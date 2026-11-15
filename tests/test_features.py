from src.data import generate_data, validate_data
from src.features import add_time_features, FEATURES
from src.forecast import chronological_split

def test_feature_engineering_and_split():
    df=validate_data(generate_data(days=120,stores=2,products=2,seed=3))
    engineered=add_time_features(df)
    assert set(FEATURES).issubset(engineered.columns)
    train,test=chronological_split(engineered,20)
    assert train.date.max() < test.date.min()
    assert len(test)>0
