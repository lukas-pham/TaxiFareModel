from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow
import joblib

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[CZ] [Prague] [lukas-pham] TaxiFare + 2"

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")


        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('model', LinearRegression())
        ])

        #if grid_search == False:
        #    pass
        params = {
            'model':
            [LinearRegression(),
              AdaBoostRegressor(),
              XGBRegressor()]
        }

        self.grid = GridSearchCV(pipe,
                                    param_grid=params,
                                    cv=5,
                                    scoring='neg_root_mean_squared_error')

    def run(self, grid_search=True):
        """set and train the pipeline"""
        self.set_pipeline()
        #if grid_search==True:
        pipe = self.grid.fit(self.X, self.y)
        self.pipe = pipe.best_estimator_
        self.mlflow_log_param(
            'model',
            type(self.pipe['model']).__name__)
        # else:
        #    self.pipe.fit(self.X, self.y)
        #    self.mlflow_log_param('model', 'linear')



    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)

        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse', rmse)

        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"

        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipe, 'model.joblib')


if __name__ == "__main__":
    # get data
    df = get_data(nrows=1000)
    # clean data
    df = clean_data(df)
    # set X, y
    y = df.pop('fare_amount')
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    # save
    trainer.save_model()
    # retrieve ID
    experiment_id = trainer.mlflow_experiment_id
    print(
    f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
