from external_data_cleaning import ext_cleaning
from feature_engineering import make_all_feature


def main():
    ext_cleaning('./data/external_data')
    train, test = make_all_feature('./data')
    train.to_csv('final_train_processed.csv', index=False)
    test.to_csv('final_test_processed.csv', index=False)


if __name__ == '__main__':
    main()
