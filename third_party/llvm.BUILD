# A package for LLVM releases.

package(default_visibility = ["//visibility:public"])

# Pre-compiled binary filegroups.

filegroup(
    name = "clang",
    srcs = ["bin/clang"],
)

filegroup(
    name = "libcxx_headers",
    srcs = glob([
        "include/c++/v1/**/*",
        "lib/clang/6.0.0/include/**/*",
    ]),
)

filegroup(
    name = "libcxx_libs",
    srcs = glob([
        "lib/libc++*",
    ]),
)

filegroup(
    name = "clang-format",
    srcs = ["bin/clang-format"],
)

filegroup(
    name = "ld",
    srcs = [
        "bin/ld.lld",
        "bin/lld",
    ],
)

filegroup(
    name = "llvm-as",
    srcs = [
        "bin/llvm-as",
    ],
)

filegroup(
    name = "llvm-dis",
    srcs = [
        "bin/llvm-dis",
    ],
)

filegroup(
    name = "llvm-link",
    srcs = [
        "bin/llvm-link",
    ],
)

filegroup(
    name = "opt",
    srcs = ["bin/opt"],
)

# The directory containing libraries.

filegroup(
    name = "libdir",
    srcs = ["lib"],
)

filegroup(
    name = "libs_files",
    srcs = glob([
        "lib/**/*",
    ]),
)

# A globbing of all static and dynamic LLVM libraries, for use in deps attrs.

cc_library(
    name = "libs",
    srcs = glob([
        "lib/lib*.a",
        "lib/lib*.so",
    ]),
)

# Dynamic libraries for LD_PRELOAD.

filegroup(
    name = "libclang_so",
    srcs = ["lib/libclang.so"],
)

filegroup(
    name = "liblto_so",
    srcs = ["lib/libLTO.so"],
)

# Static libraries, for us in data attrs of *_binary targets.

filegroup(
    name = "lib_clangARCMigrate",
    srcs = ["lib/libclangARCMigrate.a"],
)

filegroup(
    name = "lib_clangFrontend",
    srcs = ["lib/libclangFrontend.a"],
)

filegroup(
    name = "lib_clangFrontendTool",
    srcs = ["lib/libclangFrontendTool.a"],
)

filegroup(
    name = "lib_clangDriver",
    srcs = ["lib/libclangDriver.a"],
)

filegroup(
    name = "lib_clangSerialization",
    srcs = ["lib/libclangSerialization.a"],
)

filegroup(
    name = "lib_clangCodeGen",
    srcs = ["lib/libclangCodeGen.a"],
)

filegroup(
    name = "lib_clangParse",
    srcs = ["lib/libclangParse.a"],
)

filegroup(
    name = "lib_clangSema",
    srcs = ["lib/libclangSema.a"],
)

filegroup(
    name = "lib_clangRewriteFrontend",
    srcs = ["lib/libclangRewriteFrontend.a"],
)

filegroup(
    name = "lib_clangRewrite",
    srcs = ["lib/libclangRewrite.a"],
)

filegroup(
    name = "lib_clangStaticAnalyzerFrontend",
    srcs = ["lib/libclangStaticAnalyzerFrontend.a"],
)

filegroup(
    name = "lib_clangStaticAnalyzerCheckers",
    srcs = ["lib/libclangStaticAnalyzerCheckers.a"],
)

filegroup(
    name = "lib_clangStaticAnalyzerCore",
    srcs = ["lib/libclangStaticAnalyzerCore.a"],
)

filegroup(
    name = "lib_clangAnalysis",
    srcs = ["lib/libclangAnalysis.a"],
)

filegroup(
    name = "lib_clangEdit",
    srcs = ["lib/libclangEdit.a"],
)

filegroup(
    name = "lib_clangAST",
    srcs = ["lib/libclangAST.a"],
)

filegroup(
    name = "lib_clangASTMatchers",
    srcs = ["lib/libclangASTMatchers.a"],
)

filegroup(
    name = "lib_clangLex",
    srcs = ["lib/libclangLex.a"],
)

filegroup(
    name = "lib_clangBasic",
    srcs = ["lib/libclangBasic.a"],
)

filegroup(
    name = "lib_clangTooling",
    srcs = ["lib/libclangTooling.a"],
)

filegroup(
    name = "lib_clangToolingCore",
    srcs = ["lib/libclangToolingCore.a"],
)

filegroup(
    name = "lib_LLVMLTO",
    srcs = ["lib/libLLVMLTO.a"],
)

filegroup(
    name = "lib_LLVMPasses",
    srcs = ["lib/libLLVMPasses.a"],
)

filegroup(
    name = "lib_LLVMObjCARCOpts",
    srcs = ["lib/libLLVMObjCARCOpts.a"],
)

filegroup(
    name = "lib_LLVMMIRParser",
    srcs = ["lib/libLLVMMIRParser.a"],
)

filegroup(
    name = "lib_LLVMSymbolize",
    srcs = ["lib/libLLVMSymbolize.a"],
)

filegroup(
    name = "lib_LLVMDebugInfoPDB",
    srcs = ["lib/libLLVMDebugInfoPDB.a"],
)

filegroup(
    name = "lib_LLVMDebugInfoDWARF",
    srcs = ["lib/libLLVMDebugInfoDWARF.a"],
)

filegroup(
    name = "lib_LLVMCoverage",
    srcs = ["lib/libLLVMCoverage.a"],
)

filegroup(
    name = "lib_LLVMTableGen",
    srcs = ["lib/libLLVMTableGen.a"],
)

filegroup(
    name = "lib_LLVMDlltoolDriver",
    srcs = ["lib/libLLVMDlltoolDriver.a"],
)

filegroup(
    name = "lib_LLVMOrcJIT",
    srcs = ["lib/libLLVMOrcJIT.a"],
)

filegroup(
    name = "lib_LLVMXCoreDisassembler",
    srcs = ["lib/libLLVMXCoreDisassembler.a"],
)

filegroup(
    name = "lib_LLVMXCoreCodeGen",
    srcs = ["lib/libLLVMXCoreCodeGen.a"],
)

filegroup(
    name = "lib_LLVMXCoreDesc",
    srcs = ["lib/libLLVMXCoreDesc.a"],
)

filegroup(
    name = "lib_LLVMXCoreInfo",
    srcs = ["lib/libLLVMXCoreInfo.a"],
)

filegroup(
    name = "lib_LLVMXCoreAsmPrinter",
    srcs = ["lib/libLLVMXCoreAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMSystemZDisassembler",
    srcs = ["lib/libLLVMSystemZDisassembler.a"],
)

filegroup(
    name = "lib_LLVMSystemZCodeGen",
    srcs = ["lib/libLLVMSystemZCodeGen.a"],
)

filegroup(
    name = "lib_LLVMSystemZAsmParser",
    srcs = ["lib/libLLVMSystemZAsmParser.a"],
)

filegroup(
    name = "lib_LLVMSystemZDesc",
    srcs = ["lib/libLLVMSystemZDesc.a"],
)

filegroup(
    name = "lib_LLVMSystemZInfo",
    srcs = ["lib/libLLVMSystemZInfo.a"],
)

filegroup(
    name = "lib_LLVMSystemZAsmPrinter",
    srcs = ["lib/libLLVMSystemZAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMSparcDisassembler",
    srcs = ["lib/libLLVMSparcDisassembler.a"],
)

filegroup(
    name = "lib_LLVMSparcCodeGen",
    srcs = ["lib/libLLVMSparcCodeGen.a"],
)

filegroup(
    name = "lib_LLVMSparcAsmParser",
    srcs = ["lib/libLLVMSparcAsmParser.a"],
)

filegroup(
    name = "lib_LLVMSparcDesc",
    srcs = ["lib/libLLVMSparcDesc.a"],
)

filegroup(
    name = "lib_LLVMSparcInfo",
    srcs = ["lib/libLLVMSparcInfo.a"],
)

filegroup(
    name = "lib_LLVMSparcAsmPrinter",
    srcs = ["lib/libLLVMSparcAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMPowerPCDisassembler",
    srcs = ["lib/libLLVMPowerPCDisassembler.a"],
)

filegroup(
    name = "lib_LLVMPowerPCCodeGen",
    srcs = ["lib/libLLVMPowerPCCodeGen.a"],
)

filegroup(
    name = "lib_LLVMPowerPCAsmParser",
    srcs = ["lib/libLLVMPowerPCAsmParser.a"],
)

filegroup(
    name = "lib_LLVMPowerPCDesc",
    srcs = ["lib/libLLVMPowerPCDesc.a"],
)

filegroup(
    name = "lib_LLVMPowerPCInfo",
    srcs = ["lib/libLLVMPowerPCInfo.a"],
)

filegroup(
    name = "lib_LLVMPowerPCAsmPrinter",
    srcs = ["lib/libLLVMPowerPCAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMNVPTXCodeGen",
    srcs = ["lib/libLLVMNVPTXCodeGen.a"],
)

filegroup(
    name = "lib_LLVMNVPTXDesc",
    srcs = ["lib/libLLVMNVPTXDesc.a"],
)

filegroup(
    name = "lib_LLVMNVPTXInfo",
    srcs = ["lib/libLLVMNVPTXInfo.a"],
)

filegroup(
    name = "lib_LLVMNVPTXAsmPrinter",
    srcs = ["lib/libLLVMNVPTXAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMMSP430CodeGen",
    srcs = ["lib/libLLVMMSP430CodeGen.a"],
)

filegroup(
    name = "lib_LLVMMSP430Desc",
    srcs = ["lib/libLLVMMSP430Desc.a"],
)

filegroup(
    name = "lib_LLVMMSP430Info",
    srcs = ["lib/libLLVMMSP430Info.a"],
)

filegroup(
    name = "lib_LLVMMSP430AsmPrinter",
    srcs = ["lib/libLLVMMSP430AsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMMipsDisassembler",
    srcs = ["lib/libLLVMMipsDisassembler.a"],
)

filegroup(
    name = "lib_LLVMMipsCodeGen",
    srcs = ["lib/libLLVMMipsCodeGen.a"],
)

filegroup(
    name = "lib_LLVMMipsAsmParser",
    srcs = ["lib/libLLVMMipsAsmParser.a"],
)

filegroup(
    name = "lib_LLVMMipsDesc",
    srcs = ["lib/libLLVMMipsDesc.a"],
)

filegroup(
    name = "lib_LLVMMipsInfo",
    srcs = ["lib/libLLVMMipsInfo.a"],
)

filegroup(
    name = "lib_LLVMMipsAsmPrinter",
    srcs = ["lib/libLLVMMipsAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMLanaiDisassembler",
    srcs = ["lib/libLLVMLanaiDisassembler.a"],
)

filegroup(
    name = "lib_LLVMLanaiCodeGen",
    srcs = ["lib/libLLVMLanaiCodeGen.a"],
)

filegroup(
    name = "lib_LLVMLanaiAsmParser",
    srcs = ["lib/libLLVMLanaiAsmParser.a"],
)

filegroup(
    name = "lib_LLVMLanaiDesc",
    srcs = ["lib/libLLVMLanaiDesc.a"],
)

filegroup(
    name = "lib_LLVMLanaiAsmPrinter",
    srcs = ["lib/libLLVMLanaiAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMLanaiInfo",
    srcs = ["lib/libLLVMLanaiInfo.a"],
)

filegroup(
    name = "lib_LLVMHexagonDisassembler",
    srcs = ["lib/libLLVMHexagonDisassembler.a"],
)

filegroup(
    name = "lib_LLVMHexagonCodeGen",
    srcs = ["lib/libLLVMHexagonCodeGen.a"],
)

filegroup(
    name = "lib_LLVMHexagonAsmParser",
    srcs = ["lib/libLLVMHexagonAsmParser.a"],
)

filegroup(
    name = "lib_LLVMHexagonDesc",
    srcs = ["lib/libLLVMHexagonDesc.a"],
)

filegroup(
    name = "lib_LLVMHexagonInfo",
    srcs = ["lib/libLLVMHexagonInfo.a"],
)

filegroup(
    name = "lib_LLVMBPFDisassembler",
    srcs = ["lib/libLLVMBPFDisassembler.a"],
)

filegroup(
    name = "lib_LLVMBPFCodeGen",
    srcs = ["lib/libLLVMBPFCodeGen.a"],
)

filegroup(
    name = "lib_LLVMBPFAsmParser",
    srcs = ["lib/libLLVMBPFAsmParser.a"],
)

filegroup(
    name = "lib_LLVMBPFDesc",
    srcs = ["lib/libLLVMBPFDesc.a"],
)

filegroup(
    name = "lib_LLVMBPFInfo",
    srcs = ["lib/libLLVMBPFInfo.a"],
)

filegroup(
    name = "lib_LLVMBPFAsmPrinter",
    srcs = ["lib/libLLVMBPFAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMARMDisassembler",
    srcs = ["lib/libLLVMARMDisassembler.a"],
)

filegroup(
    name = "lib_LLVMARMCodeGen",
    srcs = ["lib/libLLVMARMCodeGen.a"],
)

filegroup(
    name = "lib_LLVMARMAsmParser",
    srcs = ["lib/libLLVMARMAsmParser.a"],
)

filegroup(
    name = "lib_LLVMARMDesc",
    srcs = ["lib/libLLVMARMDesc.a"],
)

filegroup(
    name = "lib_LLVMARMInfo",
    srcs = ["lib/libLLVMARMInfo.a"],
)

filegroup(
    name = "lib_LLVMARMAsmPrinter",
    srcs = ["lib/libLLVMARMAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMARMUtils",
    srcs = ["lib/libLLVMARMUtils.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUDisassembler",
    srcs = ["lib/libLLVMAMDGPUDisassembler.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUCodeGen",
    srcs = ["lib/libLLVMAMDGPUCodeGen.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUAsmParser",
    srcs = ["lib/libLLVMAMDGPUAsmParser.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUDesc",
    srcs = ["lib/libLLVMAMDGPUDesc.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUInfo",
    srcs = ["lib/libLLVMAMDGPUInfo.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUAsmPrinter",
    srcs = ["lib/libLLVMAMDGPUAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMAMDGPUUtils",
    srcs = ["lib/libLLVMAMDGPUUtils.a"],
)

filegroup(
    name = "lib_LLVMAArch64Disassembler",
    srcs = ["lib/libLLVMAArch64Disassembler.a"],
)

filegroup(
    name = "lib_LLVMAArch64CodeGen",
    srcs = ["lib/libLLVMAArch64CodeGen.a"],
)

filegroup(
    name = "lib_LLVMAArch64AsmParser",
    srcs = ["lib/libLLVMAArch64AsmParser.a"],
)

filegroup(
    name = "lib_LLVMAArch64Desc",
    srcs = ["lib/libLLVMAArch64Desc.a"],
)

filegroup(
    name = "lib_LLVMAArch64Info",
    srcs = ["lib/libLLVMAArch64Info.a"],
)

filegroup(
    name = "lib_LLVMAArch64AsmPrinter",
    srcs = ["lib/libLLVMAArch64AsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMAArch64Utils",
    srcs = ["lib/libLLVMAArch64Utils.a"],
)

filegroup(
    name = "lib_LLVMObjectYAML",
    srcs = ["lib/libLLVMObjectYAML.a"],
)

filegroup(
    name = "lib_LLVMLibDriver",
    srcs = ["lib/libLLVMLibDriver.a"],
)

filegroup(
    name = "lib_LLVMOption",
    srcs = ["lib/libLLVMOption.a"],
)

filegroup(
    name = "lib_LLVMWindowsManifest",
    srcs = ["lib/libLLVMWindowsManifest.a"],
)

filegroup(
    name = "lib_LLVMFuzzMutate",
    srcs = ["lib/libLLVMFuzzMutate.a"],
)

filegroup(
    name = "lib_LLVMX86Disassembler",
    srcs = ["lib/libLLVMX86Disassembler.a"],
)

filegroup(
    name = "lib_LLVMX86AsmParser",
    srcs = ["lib/libLLVMX86AsmParser.a"],
)

filegroup(
    name = "lib_LLVMX86CodeGen",
    srcs = ["lib/libLLVMX86CodeGen.a"],
)

filegroup(
    name = "lib_LLVMGlobalISel",
    srcs = ["lib/libLLVMGlobalISel.a"],
)

filegroup(
    name = "lib_LLVMSelectionDAG",
    srcs = ["lib/libLLVMSelectionDAG.a"],
)

filegroup(
    name = "lib_LLVMAsmPrinter",
    srcs = ["lib/libLLVMAsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMDebugInfoCodeView",
    srcs = ["lib/libLLVMDebugInfoCodeView.a"],
)

filegroup(
    name = "lib_LLVMDebugInfoMSF",
    srcs = ["lib/libLLVMDebugInfoMSF.a"],
)

filegroup(
    name = "lib_LLVMX86Desc",
    srcs = ["lib/libLLVMX86Desc.a"],
)

filegroup(
    name = "lib_LLVMMCDisassembler",
    srcs = ["lib/libLLVMMCDisassembler.a"],
)

filegroup(
    name = "lib_LLVMX86Info",
    srcs = ["lib/libLLVMX86Info.a"],
)

filegroup(
    name = "lib_LLVMX86AsmPrinter",
    srcs = ["lib/libLLVMX86AsmPrinter.a"],
)

filegroup(
    name = "lib_LLVMX86Utils",
    srcs = ["lib/libLLVMX86Utils.a"],
)

filegroup(
    name = "lib_LLVMMCJIT",
    srcs = ["lib/libLLVMMCJIT.a"],
)

filegroup(
    name = "lib_LLVMLineEditor",
    srcs = ["lib/libLLVMLineEditor.a"],
)

filegroup(
    name = "lib_LLVMInterpreter",
    srcs = ["lib/libLLVMInterpreter.a"],
)

filegroup(
    name = "lib_LLVMExecutionEngine",
    srcs = ["lib/libLLVMExecutionEngine.a"],
)

filegroup(
    name = "lib_LLVMRuntimeDyld",
    srcs = ["lib/libLLVMRuntimeDyld.a"],
)

filegroup(
    name = "lib_LLVMCodeGen",
    srcs = ["lib/libLLVMCodeGen.a"],
)

filegroup(
    name = "lib_LLVMTarget",
    srcs = ["lib/libLLVMTarget.a"],
)

filegroup(
    name = "lib_LLVMCoroutines",
    srcs = ["lib/libLLVMCoroutines.a"],
)

filegroup(
    name = "lib_LLVMipo",
    srcs = ["lib/libLLVMipo.a"],
)

filegroup(
    name = "lib_LLVMInstrumentation",
    srcs = ["lib/libLLVMInstrumentation.a"],
)

filegroup(
    name = "lib_LLVMVectorize",
    srcs = ["lib/libLLVMVectorize.a"],
)

filegroup(
    name = "lib_LLVMScalarOpts",
    srcs = ["lib/libLLVMScalarOpts.a"],
)

filegroup(
    name = "lib_LLVMLinker",
    srcs = ["lib/libLLVMLinker.a"],
)

filegroup(
    name = "lib_LLVMIRReader",
    srcs = ["lib/libLLVMIRReader.a"],
)

filegroup(
    name = "lib_LLVMAsmParser",
    srcs = ["lib/libLLVMAsmParser.a"],
)

filegroup(
    name = "lib_LLVMInstCombine",
    srcs = ["lib/libLLVMInstCombine.a"],
)

filegroup(
    name = "lib_LLVMTransformUtils",
    srcs = ["lib/libLLVMTransformUtils.a"],
)

filegroup(
    name = "lib_LLVMBitWriter",
    srcs = ["lib/libLLVMBitWriter.a"],
)

filegroup(
    name = "lib_LLVMAnalysis",
    srcs = ["lib/libLLVMAnalysis.a"],
)

filegroup(
    name = "lib_LLVMProfileData",
    srcs = ["lib/libLLVMProfileData.a"],
)

filegroup(
    name = "lib_LLVMObject",
    srcs = ["lib/libLLVMObject.a"],
)

filegroup(
    name = "lib_LLVMMCParser",
    srcs = ["lib/libLLVMMCParser.a"],
)

filegroup(
    name = "lib_LLVMMC",
    srcs = ["lib/libLLVMMC.a"],
)

filegroup(
    name = "lib_LLVMBitReader",
    srcs = ["lib/libLLVMBitReader.a"],
)

filegroup(
    name = "lib_LLVMCore",
    srcs = ["lib/libLLVMCore.a"],
)

filegroup(
    name = "lib_LLVMBinaryFormat",
    srcs = ["lib/libLLVMBinaryFormat.a"],
)

filegroup(
    name = "lib_LLVMSupport",
    srcs = ["lib/libLLVMSupport.a"],
)

filegroup(
    name = "lib_LLVMDemangle",
    srcs = ["lib/libLLVMDemangle.a"],
)
