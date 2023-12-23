//=== MapperJITLinkMemoryManager.cpp - Memory management with MemoryMapper ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Process.h"

using namespace llvm::jitlink;

namespace llvm {
namespace orc {

class MapperJITLinkMemoryManager::InFlightAlloc
    : public JITLinkMemoryManager::InFlightAlloc {
public:
  InFlightAlloc(MapperJITLinkMemoryManager &Parent, LinkGraph &G,
                ExecutorAddr AllocAddr,
                std::vector<MemoryMapper::AllocInfo::SegInfo> Segs)
      : Parent(Parent), G(G), AllocAddr(AllocAddr), Segs(std::move(Segs)) {}

  void finalize(OnFinalizedFunction OnFinalize) override {
    MemoryMapper::AllocInfo AI;
    AI.MappingBase = AllocAddr;

    std::swap(AI.Segments, Segs);
    std::swap(AI.Actions, G.allocActions());

    Parent.Mapper->initialize(AI, [OnFinalize = std::move(OnFinalize)](
                                      Expected<ExecutorAddr> Result) mutable {
      if (!Result) {
        OnFinalize(Result.takeError());
        return;
      }

      OnFinalize(FinalizedAlloc(*Result));
    });
  }

  void abandon(OnAbandonedFunction OnFinalize) override {
    Parent.Mapper->release({AllocAddr}, std::move(OnFinalize));
  }

private:
  MapperJITLinkMemoryManager &Parent;
  LinkGraph &G;
  ExecutorAddr AllocAddr;
  std::vector<MemoryMapper::AllocInfo::SegInfo> Segs;
};

MapperJITLinkMemoryManager::MapperJITLinkMemoryManager(
    size_t ReservationGranularity, std::unique_ptr<MemoryMapper> Mapper)
    : ReservationUnits(ReservationGranularity), AvailableMemory(AMAllocator),
      Mapper(std::move(Mapper)) {}

void MapperJITLinkMemoryManager::allocate(const JITLinkDylib *JD, LinkGraph &G,
                                          OnAllocatedFunction OnAllocated) {
  BasicLayout BL(G);

  // find required address space
  auto SegsSizes = BL.getSplitPageBasedLayoutSizes(Mapper->getPageSize());

  if (!SegsSizes) {
    OnAllocated(SegsSizes.takeError());
    return;
  }

  auto TotalSize = SegsSizes->total();

  auto CompleteAllocation = [this, &SegsSizes, &G, BL = std::move(BL),
                             OnAllocated = std::move(OnAllocated)](
                                Expected<ExecutorAddrRange> Result) mutable {
    if (!Result) {
      Mutex.unlock();
      return OnAllocated(Result.takeError());
    }


    auto DataSegAddr = Result->Start;
    ExecutorAddr TextSegAddr(alignDown(Result->End.getValue() - SegsSizes->TextSegs, Mapper->getPageSize()));
    auto FinalizeSegAddr =  Result->Start + alignTo(SegsSizes->DataSegs,Mapper->getPageSize());
    auto FinalizeSegAddrInit = FinalizeSegAddr;
    auto TextSegAddrInit = TextSegAddr;
    assert((FinalizeSegAddr + SegsSizes->FinalizeSegs) < (TextSegAddr, Mapper->getPageSize()) && "Not enough memory in the slab");
    std::vector<MemoryMapper::AllocInfo::SegInfo> SegInfos;

    for (auto &KV : BL.segments()) {
      auto &AG = KV.first;
      auto &Seg = KV.second;
      auto TotalSize = Seg.ContentSize + Seg.ZeroFillSize;

      ExecutorAddr *CurrAddr;
      if (AG.getMemDeallocPolicy() == orc::MemDeallocPolicy::Standard) {
          if ((AG.getMemProt() & orc::MemProt::Exec) != orc::MemProt::None) {
              CurrAddr = &TextSegAddr;
          } else {
              CurrAddr = &DataSegAddr;
          }
      } else {
          CurrAddr = &FinalizeSegAddr;
      }

      Seg.Addr = *CurrAddr;
      Seg.WorkingMem = Mapper->prepare(*CurrAddr, TotalSize);
      *CurrAddr += alignTo(TotalSize, Mapper->getPageSize());

      MemoryMapper::AllocInfo::SegInfo SI;
      SI.Offset = Seg.Addr - Result->Start;
      SI.ContentSize = Seg.ContentSize;
      SI.ZeroFillSize = Seg.ZeroFillSize;
      SI.AG = AG;
      SI.WorkingMem = Seg.WorkingMem;
      SegInfos.push_back(SI);
    }
    assert(DataSegAddr < FinalizeSegAddrInit && "Data overwrote the finalize segment");
    assert(FinalizeSegAddr < TextSegAddrInit && "Finalize overwrote the text segment");
    assert(TextSegAddr < Result->End && "Text overwrote the end of the slab");

    UsedMemory.insert({Result->Start, FinalizeSegAddr - Result->Start});
    UsedMemory.insert({TextSegAddrInit, Result->End - TextSegAddrInit});
    if (FinalizeSegAddr < TextSegAddrInit) {
      // Save the remaining memory for reuse in next allocation(s)
      AvailableMemory.insert(FinalizeSegAddr, TextSegAddrInit - 1, true);
    }
    Mutex.unlock();

    if (auto Err = BL.apply()) {
      OnAllocated(std::move(Err));
      return;
    }

    OnAllocated(std::make_unique<InFlightAlloc>(*this, G, Result->Start,
                                                std::move(SegInfos)));
  };

  Mutex.lock();

  // find an already reserved range that is large enough
  ExecutorAddrRange SelectedRange{};

  for (AvailableMemoryMap::iterator It = AvailableMemory.begin();
       It != AvailableMemory.end(); It++) {
    if (It.stop() - It.start() + 1 >= TotalSize) {
      SelectedRange = ExecutorAddrRange(It.start(), It.stop() + 1);
      It.erase();
      break;
    }
  }

  if (SelectedRange.empty()) { // no already reserved range was found
    auto TotalAllocation = alignTo(TotalSize, ReservationUnits);
    Mapper->reserve(TotalAllocation, std::move(CompleteAllocation));
  } else {
    CompleteAllocation(SelectedRange);
  }
}

void MapperJITLinkMemoryManager::deallocate(
    std::vector<FinalizedAlloc> Allocs, OnDeallocatedFunction OnDeallocated) {
  std::vector<ExecutorAddr> Bases;
  Bases.reserve(Allocs.size());
  for (auto &FA : Allocs) {
    ExecutorAddr Addr = FA.getAddress();
    Bases.push_back(Addr);
  }

  Mapper->deinitialize(Bases, [this, Allocs = std::move(Allocs),
                               OnDeallocated = std::move(OnDeallocated)](
                                  llvm::Error Err) mutable {
    // TODO: How should we treat memory that we fail to deinitialize?
    // We're currently bailing out and treating it as "burned" -- should we
    // require that a failure to deinitialize still reset the memory so that
    // we can reclaim it?
    if (Err) {
      for (auto &FA : Allocs)
        FA.release();
      OnDeallocated(std::move(Err));
      return;
    }

    {
      std::lock_guard<std::mutex> Lock(Mutex);

      for (auto &FA : Allocs) {
        ExecutorAddr Addr = FA.getAddress();
        ExecutorAddrDiff Size = UsedMemory[Addr];

        UsedMemory.erase(Addr);
        AvailableMemory.insert(Addr, Addr + Size - 1, true);

        FA.release();
      }
    }

    OnDeallocated(Error::success());
  });
}

} // end namespace orc
} // end namespace llvm
