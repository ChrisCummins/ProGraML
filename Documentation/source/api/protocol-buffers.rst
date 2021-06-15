Protocol Buffers
================

ProGraML uses `Protocol Buffers
<https://developers.google.com/protocol-buffers>`_ for representing the
structured data of program graphs.

.. contents:: Document contents:
   :local:

The Program Graph
-----------------

These protocol buffer definitions are available in:

* Python: :code:`from programl.proto import *`
* C++: :code:`#include "programl/proto/program_graph.pb.h"`

.. doxygenstruct:: ProgramGraph
   :members:

.. doxygenstruct:: Node
   :members:

.. doxygenstruct:: Edge
   :members:

.. doxygenstruct:: Function
   :members:

.. doxygenstruct:: Module
   :members:

Features
--------

These protocol buffer definitions are available in:

* Python: :code:`from programl.proto import *`
* C++: :code:`#include "programl/third_party/tesnroflow/features.pb.h"`

.. doxygenstruct:: Feature
   :members:

.. doxygenstruct:: FeatureList
   :members:

.. doxygenstruct:: FeatureLists
   :members:

.. doxygenstruct:: BytesList
   :members:

.. doxygenstruct:: Features
   :members:

.. doxygenstruct:: Features
   :members:


Util
----

These protocol buffer definitions are available in:

* Python: :code:`from programl.proto import *`
* C++: :code:`#include "programl/proto/util.pb.h"`

.. doxygenstruct:: ProgramGraphOptions
   :members:

.. doxygenstruct:: ProgramGraphList
   :members:

.. doxygenstruct:: ProgramGraphFeatures
   :members:

.. doxygenstruct:: ProgramGraphFeaturesList
   :members:

.. doxygenstruct:: Ir
   :members:

.. doxygenstruct:: IrList
   :members:

.. doxygenstruct:: SourceFile
   :members:

.. doxygenstruct:: Repo
   :members:

.. doxygenstruct:: NodeIndexList
   :members:
