# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decorators for defining components via Python functions.

Experimental: no backwards compatibility guarantees.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import types
from typing import Any, Dict, List, Text

# Standard Imports

import absl
import six

from tfx import types as tfx_types
from tfx.components.base.base_component import _SimpleComponent
from tfx.components.base.base_executor import BaseExecutor
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.components.base.function_parser import ArgFormats
from tfx.components.base.function_parser import parse_typehint_component_function
from tfx.types.component_spec import ChannelParameter


class _FunctionExecutor(BaseExecutor):
  """Base class for function-based executors."""

  # Properties that should be overridden by subclass.
  _ARG_FORMATS = {}
  _FUNCTION = staticmethod(lambda: None)
  _RETURNED_VALUES = {}

  def Do(self, input_dict: Dict[Text, List[tfx_types.Artifact]],
         output_dict: Dict[Text, List[tfx_types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    function_args = []
    for name, arg_format in self._ARG_FORMATS:
      if arg_format == ArgFormats.INPUT_ARTIFACT:
        function_args.append(input_dict[name][0])
      elif arg_format == ArgFormats.INPUT_URI:
        function_args.append(input_dict[name][0].uri)
      elif arg_format == ArgFormats.OUTPUT_ARTIFACT:
        function_args.append(output_dict[name][0])
      elif arg_format == ArgFormats.OUTPUT_URI:
        function_args.append(output_dict[name][0].uri)
      elif arg_format == ArgFormats.ARTIFACT_VALUE:
        function_args.append(input_dict[name][0].value)
      else:
        raise ValueError('Unknown argument format: %r' % (arg_format,))

    # Call function and check returned values.
    outputs = self._FUNCTION(*function_args)
    outputs = outputs or {}
    if not isinstance(outputs, dict):
      absl.logging.warning(
          ('Expected component executor function %s to return a dict of '
           'outputs (got %r instead).') % (self._FUNCTION, outputs))
      return

    # Assign returned ValueArtifact values.
    for name in self._RETURNED_VALUES:
      if name not in outputs:
        absl.logging.warning(
            'Did not receive expected output %r as return value from '
            'component.', name)
        continue
      output_dict[name][0].value = outputs[name]


def component(func: types.FunctionType):
  """Decorator: creates a component from a typehint-annotated Python function.

  This decorator creates a component based on typehint annotations specified for
  the arguments and return value for a Python function. Specifically, function
  arguments can be annotated with the following types and associated semantics:

  * `int`, `float`, `str`, `bytes`: indicates that a primitive type value will
    be passed for this argument. This value is tracked as an `Integer`, `Float`
    `String` or `Bytes` artifact (see `tfx.types.standard_artifacts`) whose
    value is read and passed into the given Python component function.
  * `InputArtifact[ArtifactType]`: indicates that an input artifact object of
    type `ArtifactType` (deriving from `tfx.types.Artifact`) will be passed for
    this argument. This artifact is intended to be consumed as an input by this
    component (possibly reading from the path specified by its `.uri`).
  * `OutputArtifact[ArtifactType]`: indicates that an output artifact object of
    type `ArtifactType` (deriving from `tfx.types.Artifact`) will be passed for
    this argument. This artifact is intended to be emitted as an output by this
    component (and written to the path specified by its `.uri`).
  * `InputUri[ArtifactType]`: indicates that the `.uri` of the input artifact
    object of type `ArtifactType` (deriving from `tfx.types.Artifact`) will be
    passed as a string for this argument. The contents at this location are
    intended to be consumed as an input by this component.
  * `OutputUri[ArtifactType]`: indicates that the `.uri` of the output artifact
    object of type `ArtifactType` (deriving from `tfx.types.Artifact`) will be
    passed as a string for this argument. The location is intended to store
    the output emitted by this component for this artifact.

  The function to which this decorator is applied must be at the top level of
  its Python module (it may not be defined within nested classes or function
  closures).

  This is example usage of component definition using this decorator:

      from tfx.components.base.annotations import OutputDict
      from tfx.components.base.annotations import InputArtifact
      from tfx.components.base.annotations import OutputArtifact
      from tfx.components.base.decorators import component
      from tfx.types.standard_artifacts import Examples
      from tfx.types.standard_artifacts import Model

      @component
      def MyTrainerComponent(
          training_data: InputArtifact[Examples],
          model: OutputArtifact[Model],
          num_iterations: int
          ) -> OutputDict(loss=float, accuracy=float):
        '''My simple trainer component.'''

        records = read_examples(training_data.uri)
        model_obj = train_model(records, num_iterations)
        model_obj.write_to(model.uri)

        return {
          'loss': model_obj.loss,
          'accuracy': model_obj.accuracy
        }

  Equivalently, `model: OutputArtifact[Model]` and `model.uri` could be replaced
  with `model_uri: OutputUri[Model]` and `model_uri`.

  Experimental: no backwards compatibility guarantees.

  Args:
    func: Typehint-annotated component executor function.

  Returns:
    BaseComponent subclass for the given component executor function.

  Raises:
    EnvironmentError: if the current Python interpreter is not Python 3.
  """
  if six.PY2:
    raise EnvironmentError('`@component` is only supported in Python 3.')

  # Defining a component within a nested class or function closure causes
  # problems because in this case, the generated component classes can't be
  # referenced via their qualified module path.
  if '<locals>' in func.__qualname__.split('.'):
    raise ValueError(
        'The @component decorator can only be applied to a function defined '
        'at the module level. It cannot be used to construct a component for a '
        'function defined in a nested class or function closure.')

  inputs, outputs, arg_formats, returned_values = (
      parse_typehint_component_function(func))

  channel_inputs = {}
  channel_outputs = {}
  for key, artifact_type in inputs.items():
    channel_inputs[key] = ChannelParameter(type=artifact_type)
  for key, artifact_type in outputs.items():
    channel_outputs[key] = ChannelParameter(type=artifact_type)
  component_spec = type(
      '%s_Spec' % func.__name__,
      (tfx_types.ComponentSpec,),
      {
          'INPUTS': channel_inputs,
          'OUTPUTS': channel_outputs,
          # TODO(ccy): add support for execution properties or remove
          # execution properties from the SDK, merging them with component
          # inputs.
          'PARAMETERS': {},
      })

  executor_class = type(
      '%s_Executor' % func.__name__,
      (_FunctionExecutor,),
      {
          '_ARG_FORMATS': arg_formats,
          # The function needs to be marked with `staticmethod` so that later
          # references of `self._FUNCTION` do not result in a bound method (i.e.
          # one with `self` as its first parameter).
          '_FUNCTION': staticmethod(func),
          '_RETURNED_VALUES': returned_values,
          '__module__': func.__module__,
      })

  # Expose the generated executor class in the same module as the decorated
  # function. This is needed so that the executor class can be accessed at the
  # proper module path. One place this is needed is in the Dill pickler used by
  # Apache Beam serialization.
  module = sys.modules[func.__module__]
  setattr(module, '%s_Executor' % func.__name__, executor_class)

  executor_spec = ExecutorClassSpec(executor_class=executor_class)

  return type(
      func.__name__, (_SimpleComponent,), {
          'SPEC_CLASS': component_spec,
          'EXECUTOR_SPEC': executor_spec,
          '__module__': func.__module__,
      })
