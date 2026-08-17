; RUN: opt -passes=sroa -S %s | FileCheck %s

; Do not convert a memset byte pattern to a non-integral pointer when
; vector-promoting an alloca partition.

; CHECK-LABEL: define <2 x ptr addrspace(10)> @vector_memset(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint

target datalayout = "e-ni:10"

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define <2 x ptr addrspace(10)> @vector_memset(<2 x ptr addrspace(10)> %v, i8 %byte) {
  %a = alloca [2 x ptr addrspace(10)], align 8
  store <2 x ptr addrspace(10)> %v, ptr %a, align 8
  %p = getelementptr inbounds i8, ptr %a, i64 8
  call void @llvm.memset.p0.i64(ptr %p, i8 %byte, i64 8, i1 false)
  %r = load <2 x ptr addrspace(10)>, ptr %a, align 8
  ret <2 x ptr addrspace(10)> %r
}
