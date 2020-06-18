import pandas as pd
import numpy as np
from . import ds_utils


import json
import requests
import bs4
BASE_URL = "https://api.weekendesk.com/api/weekend/{}/detail.json?locale=es_ES&client=msite&fetchPageModel=true"

col_names = """
package_id weekend_id country_id region_id deparment_id city_id activity_type room_type room_category trainstation_distance airport_distance
room_view room_size hotel_id address_country star located_in_seaside located_in_mountain located_in_countryside
located_in_city hotel_latitude hotel_longitude hotel_type hotel_language hotel_typologie_insolite
hotel_typologie_design hotel_typologie_charming hotel_typologie_luxury hotel_typologie_golf
hotel_typologie_spa publish_be publish_de publish_es publish_it publish_nl publish_fr free_breakfast
""".split()


def scrap(wed_id):
    url = BASE_URL.format(wed_id)
    print('getting resource:' + url, end='  ')
    response = requests.get(url)
    if response.ok:
        print("Sucess")

        soup = bs4.BeautifulSoup(response.text, "lxml")
        # this is legacy version
        # raw_data = soup.findAll('script',  {"src":False}, type="text/javascript")[1].text
        # this is maxibus version
        # raw_data = soup.find('script',  {"id":"preloaded-state"}).text
        # raw_data = raw_data.replace('window.PRELOADED_STATE =', '')

        # this is directly call api
        raw_data = response.text

        json_data = json.loads(raw_data)
        # wed_json = json_data['weekend']['loadedWeekends'][str(wed_id)]

        return json_data
    else:
        # Status Code 404, no weekend found, so it's not available
        print("Failed! Status code:,", response.status_code)


def get_data_from_web(wed_id):
    js = scrap(wed_id)
    data = {}

    data['weekend_id'] = wed_id
    data['title'] = js['pageModel']['pageTitle']
    data['price'] = js['price']['sellPrice']
    data['promotion'] = js['price']['promoPercentageRounded']
    data['uri'] = 'https://www.weekendesk.es' + js['uri']
    data['hotel_id'] = js['hotel']['id']
    data['hotel_star'] = js['hotel']['star']
    data['hotel_name'] = js['hotel']['label']
    if 'review' in js['hotel']:
        data['hotel_review_average'] = js['hotel']['review']['average']
        data['hotel_review_count'] = js['hotel']['review']['count']
    else:
        data['hotel_review_average'] = '?'
        data['hotel_review_count'] = 0
    data['latitude'] = js['hotel']['location']['latitude']
    data['longitude'] = js['hotel']['location']['longitude']
    data['image'] = js['image'][0]['url']
    # data['image'] = js['images'][0]['sizes']['960_540']
    if 'activities' in js:
        data['categories'] = [a['title'] for a in js['activities']]
    else:
        data['categories'] = []

    return data


def build_L_matrix(df, today, T, P, decay=False, weight=None):
    L = df[['user_id', 'item_id']]

    for e in df.event.unique():
        L[e] = 0
        L.loc[df[df['event' ]==e].index, e] = df.loc[df[df['event' ]==e].index, 'event_value']

    if weight:
        L.loc[L[L.view > 0].index, 'view'] = 1
        L['weight'] = df.event.map(weight)

    if decay:
        L['elapsed_time'] = (today - (pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None))).dt.days
        L['decay'] = 1/ np.exp(
            L['elapsed_time'] / T)  # L.apply(lambda row: 1/np.exp(row['elapsed_time'] / T ), axis=1)
        L['affinity'] = L['decay'] * L['weight']
    else:
        L['affinity'] = L['weight']
    L = L.groupby(['user_id', 'item_id'])['affinity'].sum().reset_index()
    L['max_affinity'] = L.groupby(['user_id'])['affinity'].transform(max)
    L['affinity'] = 1 - 1 / (np.exp(L['affinity'] / L['max_affinity'] * P))

    return L


def build_taste_matrix(df, today, T, P, package_weight_df, taste_list, decay=False, weight=None):
    L = df[['user_id', 'item_id']]

    for e in df.event.unique():
        L[e] = 0
        L.loc[df[df['event'] == e].index, e] = df.loc[df[df['event'] == e].index, 'event_value']

    if weight:
        L.loc[L[L.view > 0].index, 'view'] = 1
        L['weight'] = df.event.map(weight)

    if decay:
        L['elapsed_time'] = (today - (pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None))).dt.days
        L['decay'] = 1 / np.exp(
            L['elapsed_time'] / T)  # L.apply(lambda row: 1/np.exp(row['elapsed_time'] / T ), axis=1)
        L['affinity'] = L['decay'] * L['weight']
    else:
        L['affinity'] = L['weight']

    L = L.merge(package_weight_df, left_on='item_id', right_index=True, copy=False)
    # L.drop('item_id', axis=1, inplace=True)

    L['taste'] = L.apply(lambda row: [taste for taste in taste_list if row[taste] > 0], axis=1)
    L = L.explode('taste')
    L.dropna(inplace=True)
    L['taste_weight'] = L.apply(lambda row: row[row['taste']], axis=1)
    # L.drop(taste_list, axis=1, inplace=True)
    L['affinity'] = L['affinity'] * L['taste_weight']
    L = L.groupby(['user_id', 'taste'])['affinity'].sum().reset_index()
    L['max_affinity'] = L.groupby(['user_id'])['affinity'].transform(max)
    L['affinity'] = 1 - 1 / (np.exp(L['affinity'] / L['max_affinity'] * P))
    L.drop('max_affinity', axis=1, inplace=True)

    return L


class ItemSimilarityRecommender:
    def __init__(self, on_fly=False):
        self.on_fly=on_fly

    def load_data(self, log_path, weekend_path, S_path):

        self.log_df = pd.read_csv(log_path, dtype={'user_id': str}) if isinstance(log_path, str) else log_path
        self.S = pd.read_csv(S_path, dtype=np.uint8) if isinstance(S_path, str) else S_path
        self.weekend_df = pd.read_csv(weekend_path, header=None, names=col_names) if isinstance(weekend_path, str) else weekend_path
        self.weekend_df.drop_duplicates(subset=['package_id'], keep='last', inplace=True)

    def set_hyperparameter(self, T, P, action_weight, day):
        self.T = T
        self.P = P
        self.action_weight = action_weight
        self.today = day

    def get_recommend_packages_from_user(self, user_ids, pos=None, cookie_history=None, verbose=False):
        if not isinstance(user_ids, list):
            user_ids = [user_ids]

        if cookie_history is not None:
            self.log_df = self.log_df.append(cookie_history, ignore_index=True)

        if verbose:
            print("users:", user_ids, "looking for", pos, "packages")
        # get users history
        user_history = self.log_df[self.log_df.user_id.isin(user_ids)]

        # Build L on-the-fly
        L = build_L_matrix(user_history, today=self.today, T=self.T, P=self.P, decay=True, weight=self.action_weight)
        if verbose:
            print("L shape:", L.shape)

        # Build R matrix
        cooMat = ds_utils.AffinityMatrix(L, col_user='user_id', col_rating='affinity', col_item='item_id',
                                         col_pred='affinity')
        R = cooMat.gen_affinity_matrix()
        if verbose:
            print("R shape:", R.shape)

        # Filter S
        clicked_S_idx = [self.S.columns.get_loc(c) for c in cooMat.map_back_items.values()]
        local_S = self.S.loc[clicked_S_idx, :]
        if verbose:
            print("local S shape:", local_S.shape)

        user_package_score = np.dot(R, local_S)
        user_package_score_df = pd.DataFrame(user_package_score, index=cooMat.map_users.keys(), columns=self.S.columns)
        user_package_score_df = user_package_score_df.T

        res = {}
        for user in user_ids:
            udf = user_package_score_df[[user]]
            udf.columns = ['score']

            # get metadata
            # udf = pd.merge(udf, self.weekend_df, left_index=True, right_on="package_id")

            if pos is not None:
                # filter pos
                udf = udf[udf['publish_' + pos] == True]

            # normalisa the score
            if verbose:
                print('min score package for user', user, ':', udf.score.min())
                print('max score package for user', user, ':', udf.score.max())

            udf.score = ((udf.score - udf.score.min()) / (udf.score.max() - udf.score.min()) * 100).round(2)

            # filter viewed package
            viewed_list = user_history[user_history['user_id'] == user]['item_id'].unique()
            udf.drop(viewed_list, inplace=True, errors='ignore')

            # Sort by score
            udf = udf.sort_values('score', ascending=False).reset_index()

            res[user] = udf

        return res

    def get_recommend_packages_from_package(self, package_ids, pos=None):
        if not isinstance(package_ids, list):
            package_ids = list(package_ids)

        S_idx = [self.S.columns.get_loc(c) for c in package_ids]
        local_S = self.S.loc[S_idx, :]
        print("local S shape:", local_S.shape)

        df = local_S.T
        df['score'] = df.mean(axis=1)

        # get metadata
        df = pd.merge(df, self.weekend_df, left_index=True, right_on="package_id")
        if pos is not None:
            # filter pos
            df = df[df['publish_' + pos] == True]

        df.drop(S_idx, axis=1, inplace=True)

        # Sort by score
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        return df


class ItemAssociationRecommender:

    def load_data(self, log_path, weekend_path):
        log_df = pd.read_csv(log_path, dtype={'user_id': str})
        L = build_L_matrix(log_df, decay=True, weight=self.action_weight)
        cooMat = ds_utils.AffinityMatrix(L, col_user='user_id', col_rating='affinity', col_item='item_id',
                                         col_pred='affinity')
        R = cooMat.gen_affinity_matrix()
        R = np.clip(R, 0, 1)
        R = pd.DataFrame(R, index=cooMat.map_users.keys(), columns=cooMat.map_items.keys())
        S = np.dot(R.T, R)
        np.fill_diagonal(S, 0)
        self.S = pd.DataFrame(S, index=cooMat.map_items.keys(), columns=cooMat.map_items.keys())

        self.weekend_df = pd.read_csv(weekend_path, header=None, names=col_names)
        self.weekend_df.drop_duplicates(subset=['package_id'], keep='last', inplace=True)

    def set_hyperparameter(self, T, P, action_weight, day):
        self.T = T
        self.P = P
        self.action_weight = action_weight
        self.today = day

    def get_recommend_packages_from_package(self, package_ids, pos=None, verbose=False):
        if not isinstance(package_ids, list):
            package_ids = list(package_ids)

        # S_idx = [S.columns.get_loc(c) for c in package_ids]
        local_S = self.S.loc[package_ids, :]
        print("local S shape:", local_S.shape)

        df = local_S.T
        df['score'] = df.mean(axis=1)

        # get metadata
        df = pd.merge(df, self.weekend_df, left_index=True, right_on="package_id")
        if pos is not None:
            # filter pos
            df = df[df['publish_' + pos] == True]

        df.drop(package_ids, axis=1, inplace=True)

        # Sort by score
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        return df


class UserTasteRecommender:
    def __init__(self, T, P, action_weight, day, taste_list):
        self.T = T
        self.P = P
        self.action_weight = action_weight
        self.today = day
        self.taste_list = taste_list

    def load_data(self, log_path, weekend_path, package_path):
        self.log_df = pd.read_csv(log_path, dtype={'user_id': str}) if isinstance(log_path, str) else log_path
        self.weekend_df = pd.read_csv(weekend_path, header=None, names=col_names) if isinstance(weekend_path, str) else weekend_path
        self.weekend_df.drop_duplicates(subset=['package_id'], keep='last', inplace=True)
        package_df = pd.read_csv(package_path) if isinstance(package_path, str) else package_path
        package_df = package_df.set_index('package_id')
        self.package_df = package_df[self.taste_list]

        # Create a deep copy
        self.package_weight_df = package_df.copy()
        # Add weight
        self.package_weight_df.loc[:, :] = self.package_weight_df.div(package_df.sum(axis=0), axis=1)
        self.package_weight_df = self.package_weight_df / max(self.package_weight_df.max())

    def set_hyperparameter(self, T, P, action_weight, day, taste_list):
        self.T = T
        self.P = P
        self.action_weight = action_weight
        self.today = day
        self.taste_list = taste_list

    def get_recommend_packages_from_user(self, user_ids, pos=None, cookie_history=None, verbose=False):
        if not isinstance(user_ids, list):
            user_ids = [user_ids]

        if cookie_history is not None:
            self.log_df = self.log_df.append(cookie_history, ignore_index=True)

        if verbose:
            print("users:", user_ids, "looking for", pos, "packages")
        # get users history
        user_history = self.log_df[self.log_df.user_id.isin(user_ids)]

        # Build Taste on-the-fly
        taste = build_taste_matrix(user_history, today=self.today, T=self.T, P=self.P, package_weight_df=self.package_weight_df, taste_list=self.taste_list, decay=True, weight=self.action_weight)
        if verbose:
            print("Taste shape:", taste.shape)

        # Build R matrix
        cooMat = ds_utils.AffinityMatrix(taste, col_user='user_id', col_rating='affinity', col_item='taste',
                                         col_pred='affinity')
        R = cooMat.gen_affinity_matrix()
        if verbose:
            print("R shape:", R.shape)

        # Filter Package attributes
        local_package_df = self.package_df[cooMat.map_items.keys()]
        if verbose:
            print("local package shape:", local_package_df.shape)

        user_package_score = np.dot(R, local_package_df.T)
        user_package_score_df = pd.DataFrame(user_package_score, index=cooMat.map_users.keys(),
                                             columns=local_package_df.index)
        user_package_score_df = user_package_score_df.T

        res = {}
        for user in user_ids:
            udf = user_package_score_df[[user]]
            udf.columns = ['score']

            # get metadata
            udf = pd.merge(udf, self.weekend_df, left_index=True, right_on="package_id")

            if pos is not None:
                # filter pos
                udf = udf[udf['publish_' + pos] == True]

            # normalisa the score
            if verbose:
                print('min score package for user', user, ':', udf.score.min())
                print('max score package for user', user, ':', udf.score.max())

            udf.score = ((udf.score - udf.score.min()) / (udf.score.max() - udf.score.min()) * 100).round(2)

            # filter viewed package
            viewed_list = user_history[user_history['user_id'] == user]['item_id'].unique()
            udf.drop(viewed_list, inplace=True, errors='ignore')

            # Sort by score
            udf = udf.sort_values('score', ascending=False).reset_index(drop=True)
            res[user] = udf

        return res

