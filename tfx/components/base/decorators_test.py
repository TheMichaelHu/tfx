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
"""Tests for tfx.components.base.decorators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text
import unittest

# Standard Imports

import six

import tensorflow as tf

from tfx.components.base.annotations import OutputDict
from tfx.components.base.decorators import component
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner

if not six.PY2:
  # Currently, function components must be defined at the module level (not in
  # nested class or function scope). We define the test components here.

  @component
  def injector() -> OutputDict(a=int, b=int, c=Text, d=bytes):
    return {'a': 10, 'b': 22, 'c': 'unicode', 'd': b'bytes'}

  @component
  def simple_component(a: int, b: int, c: Text, d: bytes) -> OutputDict(
      e=float, f=float):
    del c, d
    return {'e': float(a + b), 'f': float(a * b)}

  @component
  def verify(e: float, f: float):
    assert (e, f) == (32.0, 220.0), (e, f)


@unittest.skipIf(six.PY2, 'Not compatible with Python 2.')
class ComponentDecoratorTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentDecoratorTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._metadata_path = os.path.join(self._test_dir, 'metadata.db')

  def testDefinitionInClosureFails(self):
    with self.assertRaisesRegexp(
        ValueError,
        'The @component decorator can only be applied to a function defined at '
        'the module level'):

      @component
      def my_component():  # pylint: disable=unused-variable
        return None

  def testBeamExecutionSuccess(self):
    """Test execution with return values; success case."""
    instance_1 = injector()
    instance_2 = simple_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    instance_3 = verify(e=instance_2.outputs['e'], f=instance_2.outputs['f'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testBeamExecutionFailure(self):
    """Test execution with return values; failure case."""
    instance_1 = injector()
    instance_2 = simple_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    # Swapped 'e' and 'f'.
    instance_3 = verify(e=instance_2.outputs['f'], f=instance_2.outputs['e'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    with self.assertRaisesRegexp(RuntimeError,
                                 r'AssertionError: \(220.0, 32.0\)'):
      beam_dag_runner.BeamDagRunner().run(test_pipeline)


if __name__ == '__main__':
  tf.test.main()
