from typing import NamedTuple

from kfp.v2 import compiler, dsl
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

PIPELINE_ROOT_PATH = "gs://vai-pipelines-data"
PROJECT_ID = "hackathon-team-05"
PROJECT_MODEL_NAME = "sklearn-demo"
PROJECT_SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"


@dsl.component(
    packages_to_install=["numpy", "pandas"],
    base_image="python:3.9",
    output_component_file="get_test_data.yaml"
)
def get_test_data(n_obs: int, dataset: dsl.Output[dsl.Dataset]) -> None:
    """Generate synthetic dataset."""
    from numpy.random import standard_normal
    from pandas import DataFrame

    x = standard_normal(n_obs)
    eta = standard_normal(n_obs)
    y = 0.5 * x + 0.25 * eta
    df = DataFrame({"x": x, "y": y})
    df.to_csv(dataset.path, index=False)


@dsl.component(
    packages_to_install=["joblib", "pandas", "scikit-learn"],
    base_image="python:3.9",
    output_component_file="train_model.yaml"
)
def train_model(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics]
) -> NamedTuple("output", [("deploy", str)]):
    """Train model."""
    import joblib
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv(dataset.path)
    X_train, X_test, y_train, y_test = train_test_split(dataset[["x"]], dataset["y"])

    dummy_model = DummyRegressor()
    dummy_model.fit(X_train, y_train)

    y_test_pred = dummy_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    metrics.log_metric("MAE", mae)

    model.metadata["MAE"] = mae
    joblib.dump(dummy_model, model.path)

    deploy = "true" if mae <= 1 else "false"
    return (deploy,)


@dsl.pipeline(
    name="train-and-deploy",
    pipeline_root=PIPELINE_ROOT_PATH)
def pipeline(project_id: str) -> None:
    """Train and deploy pipeline definition."""
    data_op = get_test_data(10000)

    train_model_op = train_model(data_op.outputs["dataset"])

    # model_upload_op = gcc_aip.ModelUploadOp(
    #     project_id=project_id,
    #     display_name=PROJECT_MODEL_NAME,
    #     artifact_uri = model.uri.replace("model", ""),
    #     serving_container_image_uri =  serving_container_image_uri,
    #     serving_container_health_route=f"/v1/models/{MODEL_NAME}",
    #     serving_container_predict_route=f"/v1/models/{MODEL_NAME}:predict",
    #     serving_container_environment_variables={
    #     "MODEL_NAME": MODEL_NAME,
    # },       
    #     unmanaged_container_model=train_model_op.outputs["model"],
    # )

    create_endpoint_op = gcc_aip.EndpointCreateOp(
        project=project_id,
        display_name=PROJECT_MODEL_NAME,
    )

    gcc_aip.ModelDeployOp(
        model=train_model_op.outputs["model"],
        endpoint=create_endpoint_op.outputs['endpoint'],
        automatic_resources_min_replica_count=1,
        automatic_resources_max_replica_count=1,
    )


# CI/CD
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="sklearn_train_and_deploy_demo.json")
