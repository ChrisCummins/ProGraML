
%class.java_lang_Class = type { %cdv_ty.java_lang_Class*, i8*, { %class.java_lang_reflect_Constructor*, %class.java_lang_Class*, %class.java_lang_String*, %class.java_lang_ClassLoader*, %class.java_lang_ref_SoftReference*, i32, %class.sun_reflect_generics_repository_ClassRepository*, %class.jlang_runtime_Array*, %interface.java_util_Map*, %"class.java_lang_Class$AnnotationData"*, %class.sun_reflect_annotation_AnnotationType*, %"class.java_lang_ClassValue$ClassValueMap"* } }
%cdv_ty.java_lang_Class = type opaque
%class.java_lang_reflect_Constructor = type opaque
%class.java_lang_String = type opaque
%class.java_lang_ClassLoader = type opaque
%class.java_lang_ref_SoftReference = type opaque
%class.sun_reflect_generics_repository_ClassRepository = type opaque
%class.jlang_runtime_Array = type opaque
%interface.java_util_Map = type opaque
%"class.java_lang_Class$AnnotationData" = type opaque
%class.sun_reflect_annotation_AnnotationType = type opaque
%"class.java_lang_ClassValue$ClassValueMap" = type opaque
%cdv_ty.Fib = type { %class.java_lang_Class**, i8*, { i32, [2 x i8*] }*, { %class.java_lang_Object* (%class.java_lang_Object*)*, i1 (%class.java_lang_Object*, %class.java_lang_Object*)*, void (%class.java_lang_Object*)*, %class.java_lang_Class* (%class.java_lang_Object*)*, i32 (%class.java_lang_Object*)*, void (%class.java_lang_Object*)*, void (%class.java_lang_Object*)*, %class.java_lang_String* (%class.java_lang_Object*)*, void (%class.java_lang_Object*, i64)*, void (%class.java_lang_Object*, i64, i32)*, void (%class.java_lang_Object*)*, i32 (%class.Fib*, i32)* } }
%class.java_lang_Object = type opaque
%class.Fib = type { %cdv_ty.Fib*, i8*, {} }

@Polyglot_Fib_class = global %class.java_lang_Class* null
@Polyglot_Fib_class_id = global i8 0
@Polyglot_java_lang_Object_class_id = external global i8
@Polyglot_Fib_rtti = linkonce_odr global { i32, [2 x i8*] } { i32 2, [2 x i8*] [i8* @Polyglot_Fib_class_id, i8* @Polyglot_java_lang_Object_class_id] }
@jni_JNIEnv = external global i8
@0 = private constant [4 x i8] c"fib\00"
@1 = private constant [5 x i8] c"(I)I\00"
@Polyglot_native_int = external global %class.java_lang_Class*
@2 = private global [1 x i8*] [i8* bitcast (%class.java_lang_Class** @Polyglot_native_int to i8*)]
@3 = private constant [7 x i8] c"<init>\00"
@4 = private constant [4 x i8] c"()V\00"
@5 = private global [0 x i8*] zeroinitializer
@6 = private constant [4 x i8] c"Fib\00"
@Polyglot_java_lang_Object_class = external global %class.java_lang_Class*
@Polyglot_Fib_cdv = global %cdv_ty.Fib { %class.java_lang_Class** @Polyglot_Fib_class, i8* null, { i32, [2 x i8*] }* @Polyglot_Fib_rtti, { %class.java_lang_Object* (%class.java_lang_Object*)*, i1 (%class.java_lang_Object*, %class.java_lang_Object*)*, void (%class.java_lang_Object*)*, %class.java_lang_Class* (%class.java_lang_Object*)*, i32 (%class.java_lang_Object*)*, void (%class.java_lang_Object*)*, void (%class.java_lang_Object*)*, %class.java_lang_String* (%class.java_lang_Object*)*, void (%class.java_lang_Object*, i64)*, void (%class.java_lang_Object*, i64, i32)*, void (%class.java_lang_Object*)*, i32 (%class.Fib*, i32)* } { %class.java_lang_Object* (%class.java_lang_Object*)* @Polyglot_java_lang_Object_clone__, i1 (%class.java_lang_Object*, %class.java_lang_Object*)* @Polyglot_java_lang_Object_equals__Ljava_lang_Object_2, void (%class.java_lang_Object*)* @Polyglot_java_lang_Object_finalize__, %class.java_lang_Class* (%class.java_lang_Object*)* @Polyglot_java_lang_Object_getClass__, i32 (%class.java_lang_Object*)* @Polyglot_java_lang_Object_hashCode__, void (%class.java_lang_Object*)* @Polyglot_java_lang_Object_notify__, void (%class.java_lang_Object*)* @Polyglot_java_lang_Object_notifyAll__, %class.java_lang_String* (%class.java_lang_Object*)* @Polyglot_java_lang_Object_toString__, void (%class.java_lang_Object*, i64)* @Polyglot_java_lang_Object_wait__J, void (%class.java_lang_Object*, i64, i32)* @Polyglot_java_lang_Object_wait__JI, void (%class.java_lang_Object*)* @Polyglot_java_lang_Object_wait__, i32 (%class.Fib*, i32)* @Polyglot_Fib_fib__I } }
@7 = private constant [0 x %class.java_lang_Class**] zeroinitializer
@8 = private global [0 x { i8*, i32, i32, i8*, i8* }] zeroinitializer
@9 = private global [0 x { i8*, i8*, i8*, i32, i8* }] zeroinitializer
@10 = private global [2 x { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }] [{ i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** } { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @0, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @1, i32 0, i32 0), i32 ptrtoint (i32 (%class.Fib*, i32)** getelementptr (%cdv_ty.Fib, %cdv_ty.Fib* null, i32 0, i32 3, i32 11) to i32), i8* bitcast (i32 (%class.Fib*, i32)* @Polyglot_Fib_fib__I to i8*), i8* bitcast (i32 (i8*, i64*)* @"Jni_trampoline_(LI)I" to i8*), i8* null, i32 0, i32 1, i8* bitcast (%class.java_lang_Class** @Polyglot_native_int to i8*), i32 1, i8** getelementptr inbounds ([1 x i8*], [1 x i8*]* @2, i32 0, i32 0) }, { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** } { i8* getelementptr inbounds ([7 x i8], [7 x i8]* @3, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @4, i32 0, i32 0), i32 -2, i8* bitcast (void (%class.Fib*)* @Polyglot_Fib_Fib__ to i8*), i8* bitcast (void (i8*, i64*)* @"Jni_trampoline_(L)V" to i8*), i8* null, i32 0, i32 1, i8* bitcast (%class.java_lang_Class** @Polyglot_Fib_class to i8*), i32 0, i8** getelementptr inbounds ([0 x i8*], [0 x i8*]* @5, i32 0, i32 0) }]
@Polyglot_Fib_class_info = private constant { i8*, %class.java_lang_Class**, i8*, i64, i8, i32, %class.java_lang_Class***, i32, { i8*, i32, i32, i8*, i8* }*, i32, { i8*, i8*, i8*, i32, i8* }*, i32, { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }* } { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @6, i32 0, i32 0), %class.java_lang_Class** @Polyglot_java_lang_Object_class, i8* bitcast (%cdv_ty.Fib* @Polyglot_Fib_cdv to i8*), i64 ptrtoint (%class.Fib* getelementptr (%class.Fib, %class.Fib* null, i32 1) to i64), i8 0, i32 0, %class.java_lang_Class*** getelementptr inbounds ([0 x %class.java_lang_Class**], [0 x %class.java_lang_Class**]* @7, i32 0, i32 0), i32 0, { i8*, i32, i32, i8*, i8* }* getelementptr inbounds ([0 x { i8*, i32, i32, i8*, i8* }], [0 x { i8*, i32, i32, i8*, i8* }]* @8, i32 0, i32 0), i32 0, { i8*, i8*, i8*, i32, i8* }* getelementptr inbounds ([0 x { i8*, i8*, i8*, i32, i8* }], [0 x { i8*, i8*, i8*, i32, i8* }]* @9, i32 0, i32 0), i32 2, { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }* getelementptr inbounds ([2 x { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }], [2 x { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }]* @10, i32 0, i32 0) }
@Polyglot_java_lang_Class_cdv = external global %cdv_ty.java_lang_Class

declare i8* @__GC_malloc(i64)

define linkonce_odr i32 @"Jni_trampoline_(LI)I"(i8*, i64*) !dbg !4 {
entry:
  br label %body, !dbg !9

body:                                             ; preds = %entry
  %gep = getelementptr i64, i64* %1, i32 0, !dbg !9
  %cast.arg.0 = bitcast i64* %gep to %class.java_lang_Object**, !dbg !9
  %load.arg.0 = load %class.java_lang_Object*, %class.java_lang_Object** %cast.arg.0, !dbg !9
  %gep1 = getelementptr i64, i64* %1, i32 1, !dbg !9
  %cast.arg.1 = bitcast i64* %gep1 to i32*, !dbg !9
  %load.arg.1 = load i32, i32* %cast.arg.1, !dbg !9
  %cast.func = bitcast i8* %0 to i32 (%class.java_lang_Object*, i32)*, !dbg !9
  %call = call i32 %cast.func(%class.java_lang_Object* %load.arg.0, i32 %load.arg.1), !dbg !9
  ret i32 %call, !dbg !9
}

define i32 @Polyglot_Fib_fib__I(%class.Fib*, i32) !dbg !10 {
entry:
  %x = alloca i32, !dbg !15
  br label %body, !dbg !15

body:                                             ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %x, metadata !16, metadata !17), !dbg !18
  store i32 %1, i32* %x, !dbg !15
  %load.x = load i32, i32* %x, !dbg !19
  switch i32 %load.x, label %switch.case2 [
    i32 0, label %switch.case
    i32 1, label %switch.case1
  ], !dbg !20

switch.case:                                      ; preds = %body
  ret i32 0, !dbg !21

switch.case1:                                     ; preds = %body
  ret i32 1, !dbg !22

switch.case2:                                     ; preds = %body
  %load.x3 = load i32, i32* %x, !dbg !23
  %ibinop = sub i32 %load.x3, 1, !dbg !23
  %gep = getelementptr %class.Fib, %class.Fib* %0, i32 0, i32 0, !dbg !24
  %load.dv = load %cdv_ty.Fib*, %cdv_ty.Fib** %gep, !dbg !24
  %gep4 = getelementptr %cdv_ty.Fib, %cdv_ty.Fib* %load.dv, i32 0, i32 3, i32 11, !dbg !24
  %load.dv.method = load i32 (%class.Fib*, i32)*, i32 (%class.Fib*, i32)** %gep4, !dbg !24
  %call = call i32 %load.dv.method(%class.Fib* %0, i32 %ibinop), !dbg !24
  %load.x5 = load i32, i32* %x, !dbg !25
  %ibinop6 = sub i32 %load.x5, 2, !dbg !25
  %gep7 = getelementptr %class.Fib, %class.Fib* %0, i32 0, i32 0, !dbg !26
  %load.dv8 = load %cdv_ty.Fib*, %cdv_ty.Fib** %gep7, !dbg !26
  %gep9 = getelementptr %cdv_ty.Fib, %cdv_ty.Fib* %load.dv8, i32 0, i32 3, i32 11, !dbg !26
  %load.dv.method10 = load i32 (%class.Fib*, i32)*, i32 (%class.Fib*, i32)** %gep9, !dbg !26
  %call11 = call i32 %load.dv.method10(%class.Fib* %0, i32 %ibinop6), !dbg !26
  %ibinop12 = add i32 %call, %call11, !dbg !24
  ret i32 %ibinop12, !dbg !27

switch.end:                                       ; No predecessors!
  unreachable, !dbg !15
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare %class.java_lang_Object* @Polyglot_java_lang_Object_clone__(%class.java_lang_Object*)

declare i1 @Polyglot_java_lang_Object_equals__Ljava_lang_Object_2(%class.java_lang_Object*, %class.java_lang_Object*)

declare void @Polyglot_java_lang_Object_finalize__(%class.java_lang_Object*)

declare %class.java_lang_Class* @Polyglot_java_lang_Object_getClass__(%class.java_lang_Object*)

declare i32 @Polyglot_java_lang_Object_hashCode__(%class.java_lang_Object*)

declare void @Polyglot_java_lang_Object_notify__(%class.java_lang_Object*)

declare void @Polyglot_java_lang_Object_notifyAll__(%class.java_lang_Object*)

declare %class.java_lang_String* @Polyglot_java_lang_Object_toString__(%class.java_lang_Object*)

declare void @Polyglot_java_lang_Object_wait__J(%class.java_lang_Object*, i64)

declare void @Polyglot_java_lang_Object_wait__JI(%class.java_lang_Object*, i64, i32)

declare void @Polyglot_java_lang_Object_wait__(%class.java_lang_Object*)

define linkonce_odr void @"Jni_trampoline_(L)V"(i8*, i64*) !dbg !28 {
entry:
  br label %body, !dbg !29

body:                                             ; preds = %entry
  %gep = getelementptr i64, i64* %1, i32 0, !dbg !29
  %cast.arg.0 = bitcast i64* %gep to %class.java_lang_Object**, !dbg !29
  %load.arg.0 = load %class.java_lang_Object*, %class.java_lang_Object** %cast.arg.0, !dbg !29
  %cast.func = bitcast i8* %0 to void (%class.java_lang_Object*)*, !dbg !29
  call void %cast.func(%class.java_lang_Object* %load.arg.0), !dbg !29
  ret void, !dbg !29
}

define void @Polyglot_Fib_Fib__(%class.Fib*) !dbg !30 {
entry:
  br label %body, !dbg !33

body:                                             ; preds = %entry
  %call = call %class.java_lang_Object* @getGlobalMutexObject(), !dbg !33
  call void @jni_MonitorEnter(i8* @jni_JNIEnv, %class.java_lang_Object* %call), !dbg !33
  %class = load %class.java_lang_Class*, %class.java_lang_Class** @Polyglot_Fib_class, !dbg !33
  %class.null = icmp eq %class.java_lang_Class* %class, null, !dbg !33
  br i1 %class.null, label %load.class, label %continue, !dbg !33

load.class:                                       ; preds = %body
  %call1 = call %class.java_lang_Class* @Polyglot_Fib_load_class(), !dbg !33
  br label %continue, !dbg !33

continue:                                         ; preds = %load.class, %body
  %call2 = call %class.java_lang_Object* @getGlobalMutexObject(), !dbg !33
  call void @jni_MonitorExit(i8* @jni_JNIEnv, %class.java_lang_Object* %call2), !dbg !33
  %cast.erasure = bitcast %class.Fib* %0 to %class.java_lang_Object*, !dbg !34
  call void @Polyglot_java_lang_Object_Object__(%class.java_lang_Object* %cast.erasure), !dbg !34
  ret void, !dbg !33
}

declare %class.java_lang_Object* @getGlobalMutexObject()

declare void @jni_MonitorEnter(i8*, %class.java_lang_Object*)

define %class.java_lang_Class* @Polyglot_Fib_load_class() !dbg !35 {
entry:
  br label %body, !dbg !37

body:                                             ; preds = %entry
  %call = call %class.java_lang_Object* @getGlobalMutexObject(), !dbg !37
  call void @jni_MonitorEnter(i8* @jni_JNIEnv, %class.java_lang_Object* %call), !dbg !37
  %call1 = call i8* @__GC_malloc(i64 ptrtoint (%class.java_lang_Class* getelementptr (%class.java_lang_Class, %class.java_lang_Class* null, i32 1) to i64)), !dbg !37
  %cast = bitcast i8* %call1 to %class.java_lang_Class*, !dbg !37
  store %class.java_lang_Class* %cast, %class.java_lang_Class** @Polyglot_Fib_class, !dbg !37
  %gep = getelementptr %class.java_lang_Class, %class.java_lang_Class* %cast, i32 0, i32 0, !dbg !37
  store %cdv_ty.java_lang_Class* @Polyglot_java_lang_Class_cdv, %cdv_ty.java_lang_Class** %gep, !dbg !37
  %call2 = call %class.java_lang_Object* @getGlobalMutexObject(), !dbg !37
  call void @jni_MonitorEnter(i8* @jni_JNIEnv, %class.java_lang_Object* %call2), !dbg !37
  %class = load %class.java_lang_Class*, %class.java_lang_Class** @Polyglot_java_lang_Object_class, !dbg !37
  %class.null = icmp eq %class.java_lang_Class* %class, null, !dbg !37
  br i1 %class.null, label %load.class, label %continue, !dbg !37

load.class:                                       ; preds = %body
  %call3 = call %class.java_lang_Class* @Polyglot_java_lang_Object_load_class(), !dbg !37
  br label %continue, !dbg !37

continue:                                         ; preds = %load.class, %body
  %call4 = call %class.java_lang_Object* @getGlobalMutexObject(), !dbg !37
  call void @jni_MonitorExit(i8* @jni_JNIEnv, %class.java_lang_Object* %call4), !dbg !37
  call void @RegisterJavaClass(%class.java_lang_Class* %cast, { i8*, %class.java_lang_Class**, i8*, i64, i8, i32, %class.java_lang_Class***, i32, { i8*, i32, i32, i8*, i8* }*, i32, { i8*, i8*, i8*, i32, i8* }*, i32, { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }* }* @Polyglot_Fib_class_info), !dbg !37
  %call5 = call %class.java_lang_Object* @getGlobalMutexObject(), !dbg !37
  call void @jni_MonitorExit(i8* @jni_JNIEnv, %class.java_lang_Object* %call5), !dbg !37
  ret %class.java_lang_Class* %cast, !dbg !37
}

declare void @jni_MonitorExit(i8*, %class.java_lang_Object*)

declare void @Polyglot_java_lang_Object_Object__(%class.java_lang_Object*)

declare %class.java_lang_Class* @Polyglot_java_lang_Object_load_class()

declare void @RegisterJavaClass(%class.java_lang_Class*, { i8*, %class.java_lang_Class**, i8*, i64, i8, i32, %class.java_lang_Class***, i32, { i8*, i32, i32, i8*, i8* }*, i32, { i8*, i8*, i8*, i32, i8* }*, i32, { i8*, i8*, i32, i8*, i8*, i8*, i32, i32, i8*, i32, i8** }* }*)

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_Java, file: !1, producer: "JLang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "fib.java", directory: "/mnt/c/Users/d067621/OneDrive - SAP SE/Documents/Research/IVAN")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "Jni_trampoline_(LI)I", linkageName: "Jni_trampoline_(LI)I", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8}
!7 = !DIBasicType(name: "void*", size: 64, encoding: DW_ATE_address)
!8 = !DIBasicType(name: "jvalue*", size: 64, encoding: DW_ATE_address)
!9 = !DILocation(line: 3, column: 15, scope: !4)
!10 = distinct !DISubprogram(name: "Fib#fib(int)", linkageName: "Polyglot_Fib_fib__I", scope: !1, file: !1, line: 3, type: !11, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14}
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "Fib", file: !1, line: 2)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 3, column: 15, scope: !10)
!16 = !DILocalVariable(name: "x", scope: !10, file: !1, line: 3, type: !14)
!17 = !DIExpression()
!18 = !DILocation(line: 3, column: 19, scope: !10)
!19 = !DILocation(line: 4, column: 15, scope: !10)
!20 = !DILocation(line: 4, column: 8, scope: !10)
!21 = !DILocation(line: 6, column: 16, scope: !10)
!22 = !DILocation(line: 8, column: 16, scope: !10)
!23 = !DILocation(line: 10, column: 27, scope: !10)
!24 = !DILocation(line: 10, column: 23, scope: !10)
!25 = !DILocation(line: 10, column: 40, scope: !10)
!26 = !DILocation(line: 10, column: 36, scope: !10)
!27 = !DILocation(line: 10, column: 16, scope: !10)
!28 = distinct !DISubprogram(name: "Jni_trampoline_(L)V", linkageName: "Jni_trampoline_(L)V", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!29 = !DILocation(line: 2, column: 17, scope: !28)
!30 = distinct !DISubprogram(name: "Fib#Fib()", linkageName: "Polyglot_Fib_Fib__", scope: !1, file: !1, line: 2, type: !31, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!31 = !DISubroutineType(types: !32)
!32 = !{!13}
!33 = !DILocation(line: 2, column: 17, scope: !30)
!34 = !DILocation(line: 2, column: 7, scope: !30)
!35 = distinct !DISubprogram(name: "load_Fib", linkageName: "Polyglot_Fib_load_class", scope: !1, file: !1, line: 2, type: !36, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!36 = !DISubroutineType(types: !2)
!37 = !DILocation(line: 2, column: 7, scope: !35)
