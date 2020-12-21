import pandas as pd
import numpy as np

class Information:
    def __init__(self):
        #用户、电影、评分总数
        self.user_number = 943
        self.items_number = 1682
        self.rating_number = 100000
        
    def load_info(self):
        # 读入用户信息
        user_names = ['user id', 'age', 'gender', 'occupation', 'zip code']
        self.user_info = pd.read_table('./ml-100k/u.user', sep = '\|', names = user_names, engine = 'python')
        
        # 读入用户评分
        rating_names = ['user id', 'item id', 'rating', 'timestamp']
        self.rating_info = pd.read_table('./ml-100k/u.data', sep = '\t', names = rating_names, engine = 'python')
        
        # 读入电影信息
        item_names = ['movie id', 'movie title', 'release date', 'video release date', 
                'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                'Thriller', 'War', 'Western']
        self.item_info = pd.read_table('./ml-100k/u.item', sep = '\|', names = item_names, engine = 'python')
        
        #读入电影类型编号
        genre_names = ['genre', 'index']
        self.genre_info = pd.read_table('./ml-100k/u.genre', sep = '\|', names = genre_names, engine = 'python')
        
        #读入职业
        occu_names = ['occupation']
        self.occu_info = pd.read_table('./ml-100k/u.occupation', names = occu_names, engine = 'python')
    
    def question1(self):
        '''
        计算每种类别的电影数量

        Returns
        -------
        genre_df : DataFrame
            每种类别的电影数量

        '''
        genre_df = pd.DataFrame()
        genre_df = self.item_info.loc[0:self.items_number-1, 'unknown':'Western']
        print(genre_df.sum())
        return genre_df


    def question2(self):
        '''
        计算不同职业倾向的评分高低

        Returns
        -------
        occu_rating_mean_df : DataFrame
            不同职业人群给出的评分均值
        occu_rating_var_df : DataFrame
            不同职业人群给出的评分方差

        '''
        user_df = pd.DataFrame()
        user_df['user id'] = self.user_info['user id']
        user_df['occupation'] = self.user_info['occupation']
        rating_df = pd.DataFrame()
        rating_df['user id'] = self.rating_info['user id']
        rating_df['rating'] = self.rating_info['rating']
        # 连接表
        uid_occu_rating_df = pd.merge(user_df, rating_df, on='user id')
        uid_occu_rating_df = uid_occu_rating_df[['occupation','rating']]
        # 分组
        occu_rating_df = uid_occu_rating_df.groupby("occupation")
        # 平均值
        occu_rating_mean_df = occu_rating_df.agg('mean').sort_values(by='rating',ascending = False   )
        # 方差
        occu_rating_var_df = occu_rating_df.agg('var').sort_values(by='rating')
        return occu_rating_mean_df, occu_rating_var_df
    
    def question3(self):
        '''
        不同职业喜爱的电影类型

        Returns
        -------
        None.

        '''
        user_df = pd.DataFrame()
        user_df['user id'] = self.user_info['user id']
        user_df['item id'] = self.user_info['item id']
        user_df['occupation'] = self.user_info['occupation']
        rating_df = pd.DataFrame()
        rating_df['user id'] = self.rating_info['user id']
        rating_df['rating'] = self.rating_info['rating']
        item_df = pd.DataFrame()
        item_df['item id'] = self.item_info['movie id']
        # TODO: 把不同类型电影加进来
        
    
    

if __name__ == '__main__':
    info = Information()
    info.load_info()
    
    info.question2()
    
    






