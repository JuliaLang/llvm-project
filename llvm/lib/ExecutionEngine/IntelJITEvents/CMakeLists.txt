include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )
if(NOT DEFINED ITTAPI_SOURCE_DIR)
    set(ITTAPI_SOURCE_DIR ${PROJECT_BINARY_DIR}/ittapi)
endif()
include_directories( ${ITTAPI_SOURCE_DIR}/include/ )
add_llvm_component_library(LLVMIntelJITEvents
  IntelJITEventListener.cpp


  LINK_COMPONENTS
  CodeGen
  Core
  DebugInfoDWARF
  Support
  Object
  ExecutionEngine
  IntelJITProfiling
)

add_dependencies(LLVMIntelJITEvents LLVMCodeGen)
