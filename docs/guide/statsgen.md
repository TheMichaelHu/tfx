# The StatisticsGen TFX Pipeline Component

The StatisticsGen TFX pipeline component generates features statistics
over both training and serving data, which can be used by other pipeline
components.
StatisticsGen uses Beam to scale to large datasets.

* Consumes: datasets created by an ExampleGen pipeline component.
* Emits: Dataset statistics.

## StatisticsGen and TensorFlow Data Validation

StatisticsGen makes extensive use of [TensorFlow Data Validation](tfdv.md) for
generating statistics from your dataset.

## Using the StatsGen Component

A StatisticsGen pipeline component is typically very easy to deploy and
requires little
customization. Typical code looks like this:

```python
from tfx import components

...

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## Using the StatsGen Component With a Schema

For the first run of a pipeline, the output of StatisticsGen will be used to
infer a schema. However, on subsequent runs you may have a manually curated
schema that contains additional information about your data set. By providing
this schema to StatisticsGen, TFDV can provide more useful statistics based on
declared properties of your data set.

In this setting, you will invoke StatisticsGen with a curated schema that has
been imported by an ImporterNode like this:

```python
from tfx import components
from tfx.types import standard_artifacts

...

user_schema_importer = components.ImporterNode(
    instance_name='import_user_schema',
    source_uri=user_schema_dir, # directory containing only schema text proto
    artifact_type=standard_artifacts.Schema)

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=user_schema_importer.outputs['result'],
      name='compute-eval-stats'
      )
```

### Creating a Curated Schema

To create an instance of the TensorFlow Metadata
[`Schema` proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto),
you could compose a valid
[text format](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html)
proto from scratch. However, it is probably better to use the inferred schema
produced by `SchemaGen` as a starting point. To retrieve the inferred schema
after running `SchemaGen`, modify it, and then write it to a file, you can use
the following steps:

```python
import os
from tfx.utils import io_utils
from tfx.types import artifact_utils

schema_gen = SchemaGen(...)
context.run(schema_gen)
...

schema = io_utils.SchemaReader().read(
    io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(schema_gen.outputs['schema'].get())))

# Optionally, programmatically modify schema
for feature in schema.feature:
  if feature.name == 'image':
    feature.image_domain.SetInParent()

# Write modified schema to file
user_schema_dir = '/path/to/persistent/schema/'
io_utils.write_pbtxt_file(
    os.path.join(user_schema_dir, 'schema.pbtxt'), schema)
```

Once you've got a copy of the text format `Schema` proto, you can modify it to
communicate information like whether a given feature is a
[categorical integer](https://github.com/tensorflow/metadata/blob/de3406014940900c06c05cf576d1ba0a3ea4c9ae/tensorflow_metadata/proto/v0/schema.proto#L415),
or
[raw image bytes](https://github.com/tensorflow/metadata/blob/de3406014940900c06c05cf576d1ba0a3ea4c9ae/tensorflow_metadata/proto/v0/schema.proto#L161).
This will produce more informative stats, and can be used to enforce additional
assumptions about your data with the
[`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) component.
