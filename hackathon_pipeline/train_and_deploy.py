"""
Vertex AI pipeline for the hackathon project, with the following stages:
 - get data (uses synthetic data generation);
 - train model using Scikit-Learn; and,
 - uploads model to Vertex AI registry, creates endpoint and deploys
   the model to the endpoint.

Each step is implemented within a Kubeflow component, where the final
stage uses the Google Cloud AI platform API.
"""
from kfp.v2 import compiler, dsl

PIPELINE_ROOT_PATH = "gs://vai-pipelines-data/hackathon"
PROJECT_ID = "hackathon-team-05"
PROJECT_MODEL_NAME = "nyc-public-bike-ride-forecasts"
PROJECT_SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"


@dsl.component(
    packages_to_install=["numpy", "pandas"],
    base_image="python:3.9",
    output_component_file="pipelines/build/get_test_data.yaml"
)
def get_data(dataset: dsl.Output[dsl.Artifact]) -> None:
    """Generate data from BigQuery."""
    from google.cloud import bigquery

    query_string = """
    SELECT
    CONCAT(
        'https://stackoverflow.com/questions/',
        CAST(id as STRING)) as url,
    view_count
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    WHERE tags like '%google-bigquery%'
    ORDER BY view_count DESC
    """

    dataset = (
        bigquery.Client().query(query_string)
        .result()
        .to_dataframe(create_bqstorage_client=True)
    )
    dataset.to_csv(f"{dataset.path}.csv")


@dsl.component(
    packages_to_install=["joblib", "pandas", "scikit-learn"],
    base_image="python:3.9",
    output_component_file="pipelines/build/train_model.yaml"
)
def train_model(
    dataset: dsl.Input[dsl.Artifact],
    model: dsl.Output[dsl.Artifact],
    metrics: dsl.Output[dsl.Metrics]
) -> None:
    """Train model."""
    import joblib
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv(f"{dataset.path}.csv")
    X_train, X_test, y_train, y_test = train_test_split(dataset[["x"]], dataset["y"])

    dummy_model = DummyRegressor()
    dummy_model.fit(X_train, y_train)

    y_test_pred = dummy_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    metrics.log_metric("MAE", mae)

    model.metadata["MAE"] = mae
    joblib.dump(dummy_model, f"{model.path}.joblib")


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-aiplatform']
)
def deploy(
    project_id: str,
    project_model_name: str,
    project_serving_image: str,
    model: dsl.Input[dsl.Artifact],
    vertex_endpoint: dsl.Output[dsl.Artifact],
    vertex_model: dsl.Output[dsl.Artifact]
) -> None:
    """Upload model, create endpoint and deploy!"""
    from google.cloud import aiplatform
    aiplatform.init(project=project_id)

    uploaded_model = aiplatform.Model.upload(
        display_name=project_model_name,
        artifact_uri=model.uri.replace("/model", ""),
        serving_container_image_uri=project_serving_image,
    )

    endpoint = uploaded_model.deploy(
        machine_type='n1-standard-4'
    )
    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = uploaded_model.resource_name


@dsl.pipeline(
    name="train-and-deploy",
    pipeline_root=PIPELINE_ROOT_PATH)
def pipeline(project_id: str) -> None:
    """Train and deploy pipeline definition."""
    data_op = get_data()

    train_model_op = train_model(data_op.outputs["dataset"])

    deploy(
        PROJECT_ID,
        PROJECT_MODEL_NAME,
        PROJECT_SERVING_IMAGE,
        train_model_op.outputs["model"]
    )


# example step used to create build artefacts in CI/CD pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="hackathon_pipeline/build/hackathon_train_and_deploy.json")
