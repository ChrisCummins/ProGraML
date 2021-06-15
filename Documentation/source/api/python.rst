Python API Reference
====================

.. automodule:: programl

.. contents:: Document contents:
    :local:


Graph Creation Ops
------------------

.. automodule:: programl.create_ops

.. currentmodule:: programl


LLVM / Clang
~~~~~~~~~~~~

.. autofunction:: from_cpp

.. autofunction:: from_clang

.. autofunction:: from_llvm_ir

.. autofunction:: programl.util.py.cc_system_includes.get_system_includes


XLA
~~~

.. autofunction:: from_xla_hlo_proto


.. autoexception:: GraphCreationError

.. autoexception:: UnsupportedCompiler


Graph Transform Ops
-------------------

.. automodule:: programl.transform_ops

.. currentmodule:: programl

DGL
~~~

.. autofunction:: to_dgl

NetworkX
~~~~~~~~

.. autofunction:: to_networkx

Graphviz
~~~~~~~~

.. autofunction:: to_dot

JSON
~~~~

.. autofunction:: to_json

.. autoexception:: GraphTransformError


Graph Serialization
-------------------

.. automodule:: programl.serialize_ops

.. currentmodule:: programl

File
~~~~

.. autofunction:: save_graphs

.. autofunction:: load_graphs

Byte Array
~~~~~~~~~~

.. autofunction:: to_bytes

.. autofunction:: from_bytes

String
~~~~~~

.. autofunction:: to_string

.. autofunction:: from_string
