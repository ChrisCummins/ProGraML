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


XLA
~~~

.. autofunction:: from_xla_hlo_proto


Graph Transform Ops
-------------------

.. automodule:: programl.transform_ops

.. currentmodule:: programl

.. autofunction:: to_json

.. autofunction:: to_networkx

.. autofunction:: to_dgl

.. autofunction:: to_dot


Graph Serialization
-------------------

.. automodule:: programl.serialize_ops

.. currentmodule:: programl

.. autofunction:: save_graphs

.. autofunction:: load_graphs

.. autofunction:: to_bytes

.. autofunction:: from_bytes

.. autofunction:: to_string

.. autofunction:: from_string


Errors
------

.. autoexception:: GraphCreationError

.. autoexception:: GraphTransformError

.. autoexception:: UnsupportedCompiler
