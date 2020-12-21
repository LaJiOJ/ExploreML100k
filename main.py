import pandas as pd
import numpy as np

class Information:
    def __init__(self):
        #用户、电影、评分总数
        self.user_number = 943
        self.movie_number = 1682
        self.rating_number = 100000
        
    def load_info(self):
        # 读入用户信息
        user_names = ['user id', 'age', 'gender', 'occupation', 'zip code']
        self.user_info = pd.read_table('./ml-100k/u.user', sep = '\|', names = user_names, engine = 'python')
        
        # 读入用户评分
        rating_names = ['user id', 'movie id', 'rating', 'timestamp']
        self.rating_info = pd.read_table('./ml-100k/u.data', sep = '\t', names = rating_names, engine = 'python')
        
        # 读入电影信息
        movie_names = ['movie id', 'movie title', 'release date', 'video release date', 
                'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                'Thriller', 'War', 'Western']
        self.movie_info = pd.read_table('./ml-100k/u.item', sep = '\|', names = movie_names, engine = 'python')
        
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
        genre_df = self.movie_info.loc[0:self.movie_number-1, 'unknown':'Western']
        print(genre_df.sum())
        return genre_df.sum()


    def question2(self):
        '''
        计算不同职业的评分高低

        Returns
        -------
        occu_rating_mean_df : DataFrame
            不同职业人群给出的评分均值
        occu_rating_var_df : DataFrame
            不同职业人群给出的评分方差

        '''
        user_df = pd.DataFrame(self.user_info,
                               columns=['user id', 'occupation'])
        rating_df = pd.DataFrame(self.rating_info, 
                                 columns=['user id', 'rating'])
        # 连接表
        uid_occu_rating_df = pd.merge(user_df, rating_df, on='user id')
        uid_occu_rating_df = uid_occu_rating_df[['occupation','rating']]
        # 分组
        occu_rating_df = uid_occu_rating_df.groupby('occupation')
        # # 初步展示结果
        # occu_rating_df.agg(['mean','var']).plot(kind='bar')
        # 平均值
        occu_rating_mean_df = occu_rating_df.agg('mean').sort_values(by='rating',ascending = False   )
        # 方差
        occu_rating_var_df = occu_rating_df.agg('var').sort_values(by='rating')
        return occu_rating_mean_df, occu_rating_var_df
    
    def question3(self):
        '''
        计算不同性别不同年龄段的评分

        Returns
        -------
        age_gender_rating_mean_df : DataFrame
            不同性别不同年龄段的评分平均数
        age_gender_rating_var_df : DataFrame
            不同性别不同年龄段的评分方差

        '''
        user_df = pd.DataFrame(self.user_info,
                               columns=['user id', 'age', 'gender'])
        rating_df = pd.DataFrame(self.rating_info, 
                                 columns=['user id', 'rating'])
        # 连接表
        uid_age_rating_df = pd.merge(user_df, rating_df, on='user id')
        uid_age_rating_df = uid_age_rating_df[['age','gender','rating']]
        # 年龄分组
        age_group = pd.cut(uid_age_rating_df['age'],bins=[0,18,25,35,60,100])# cut左开右闭，且数据集中没有>100岁的人
        age_gender_rating_df = uid_age_rating_df.groupby([age_group,'gender'])
        # # 初步展示结果
        # age_gender_rating_df.agg(['mean','var']).plot(kind='bar')
        # 平均值
        age_gender_rating_mean_df = age_gender_rating_df.agg('mean').sort_values(by='rating', ascending=False)['rating']
        # 方差
        age_gender_rating_var_df = age_gender_rating_df.agg('var').sort_values(by='rating')['rating']
        return age_gender_rating_mean_df, age_gender_rating_var_df
    
    def question4(self):
        '''
        不同职业喜爱的电影类型

        Returns
        -------
        None.

        '''
        user_df = pd.DataFrame(self.user_info, columns=['user id', 'occupation'])
        
        rating_df = pd.DataFrame(self.rating_info, columns=['user id', 'movie id', 'rating'])
        
        columns_need = ['movie id']
        columns_need.extend(self.movie_info.columns.tolist()[5:])
        movie_df = pd.DataFrame(self.movie_info, columns = columns_need)
        
        user_rating_df = pd.merge(user_df, rating_df, on='user id')
        
        user_rating_movie_df = pd.merge(user_rating_df,
                                       movie_df,
                                       how='left',
                                       on='movie id')
        
        # 分组
        user_rating_movie_df = user_rating_movie_df.groupby('occupation')
        
        return user_rating_movie_df
        
        
if __name__ == '__main__':
    info = Information()
    info.load_info()
    # genre_df = info.question1()
    
    # occu_rating_mean_df, occu_rating_var_df = info.question2()
    
    # age_rating_mean_df, age_rating_var_df = info.question3()
    
    user_rating_movie_df = info.question4()    






