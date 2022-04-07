"""
Vertex AI pipeline for the hackathon project, which trains a model that
predicts bike rides in NYC, depending on the weather. Contains the
following stages:
 - get data from BigQuery;
 - train model using Scikit-Learn; and,
 - upload model to Vertex AI registry, creates endpoint and deploys
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
    packages_to_install=["numpy", "pandas", "google-cloud-aiplatform", "pyarrow"],
    base_image="python:3.9",
    output_component_file="hackathon_pipeline/build/get_test_data.yaml"
)
def get_data(project_id: str, dataset: dsl.Output[dsl.Dataset]) -> None:
    """Generate data from BigQuery."""
    from google.cloud import bigquery

    query_string = """
        SELECT 
            EXTRACT(DAYOFYEAR FROM bike.starttime) AS day_of_year,
            EXTRACT(YEAR FROM bike.starttime) AS year,
            SUM(bike.tripduration) AS total_duration,
            MAX(bike.start_station_latitude) AS lat,
            MAX(bike.start_station_longitude) AS lon,
            AVG(weather.prcp) AS avg_prcp,
            AVG(weather.tmin) AS avg_tmin,
            AVG(weather.tmax) AS avg_tmax,
            AVG(weather.snow) AS avg_snow,
            AVG(weather.wind) AS avg_wind
        FROM `hackathon-team-05.team05bike.noaa_daily_weather_nyc_central_park` AS weather
        INNER JOIN `hackathon-team-05.team05bike.NYCbike` AS bike
        ON weather.date = EXTRACT(DATE FROM bike.starttime)
        GROUP BY 1, 2
    """

    df = (
        bigquery.Client(project=project_id).query(query_string)
        .result()
        .to_dataframe(create_bqstorage_client=True)
    )
    df.to_csv(f"{dataset.path}.csv")


@dsl.component(
    packages_to_install=["joblib", "pandas", "scikit-learn"],
    base_image="python:3.9",
    output_component_file="hackathon_pipeline/build/train_model.yaml"
)
def train_model(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
) -> None:
    """Train model."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
        
    def sort(df, sort_cols):
        return df.sort_values(sort_cols)

    def define_mobility_index(df, col_to_index):
        df['mobility_index'] = 100 * df[col_to_index] / df[col_to_index].iloc[0]
        return df, 'mobility_index'

    def add_pct_change_to_index(df, y, period):
        df[y + '_diff'] = df[y].pct_change(periods=period)
        return df
        
    def add_momentum_terms(df, mobility_index, rolling_averages):
        for rolling_avg in rolling_averages:
            df['momentum_r' + str(rolling_avg)] = (
                df[mobility_index + '_diff'].rolling(rolling_avg).mean() * 1 + df[mobility_index]
            ).shift(1)
        return df

    def add_timeshift_to_index(df, mobility_index, shifts_to_y):
        for shift in shifts_to_y:
            df[mobility_index + '_t' + str(abs(shift))] = df[mobility_index].shift(shift)
        return df

    def setup_df(df, sort_cols, col_to_index, period, rolling_averages, shifts_to_y):
        df = sort(df, sort_cols)
        df, mobility_index = define_mobility_index(df, col_to_index)
        df = add_pct_change_to_index(df, mobility_index, period)
        df = add_momentum_terms(df, mobility_index, rolling_averages)
        df = add_timeshift_to_index(df, mobility_index, shifts_to_y)
        df = df.dropna()
        return df

    df = setup_df(
        df=pd.read_csv(f"{dataset.path}.csv"),
        sort_cols=['year', 'day_of_year'],
        col_to_index='total_duration',
        period=1,
        rolling_averages=[2, 3, 5, 10],
        shifts_to_y=[-1]
    )

    X = [
        'mobility_index',
        'momentum_r2',
        'momentum_r3',
        'momentum_r5',
        'momentum_r10',
        'avg_prcp',
        'avg_tmin',
        'avg_tmax',
        'avg_snow',
        'avg_wind'
    ]

    y = 'mobility_index_t1'
    mask = (df['year'] == 2013) | (df['year'] == 2014) | (df['year'] == 2015) | (df['year'] == 2016)
    rfr = RandomForestRegressor(random_state=123)
    rfr.fit(df.loc[mask, X], df.loc[mask, y])
    joblib.dump(rfr, f"{model.path}.joblib")


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-aiplatform'],
    output_component_file="hackathon_pipeline/build/get_test_data.yaml"
)
def deploy(
    project_id: str,
    project_model_name: str,
    project_serving_image: str,
    model: dsl.Input[dsl.Model],
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
    name="nyc-bike-ride-forecasts-train-and-deploy",
    pipeline_root=PIPELINE_ROOT_PATH)
def pipeline(project_id: str) -> None:
    """Train and deploy pipeline definition."""
    data_op = get_data(PROJECT_ID)

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
