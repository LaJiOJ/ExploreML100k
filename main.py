import pandas as pd
import numpy as np

class Information:
    def __init__(self):
        #用户、电影、评分总数
        self.user_number = 943
        self.movie_number = 1682
        self.rating_number = 100000
        self.age_bin = [0,18,25,35,60,100]
        
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

    def question1_2(self):
        '''
        #寻找出现电影名字中最常出现的词

        Returns
        -------
        word_dct_ten : list
            最常出现的10个词
            
        '''
        movie_df = pd.DataFrame()
        movie_df = self.movie_info['movie title']
        movie_list = movie_df.tolist()
        #对数据进行基础的处理
        word_list=[]
        for i in movie_list:
            i = i.split()[:-1] #去掉末尾的年份
            for j in i:
                j = j.lower()
                j = (j.replace(',','').replace('.','').replace(':','').replace('*','').
                    replace('&','').replace('(','').replace(')',''))        
                word_list.append(j)
        #通过字典计算词频
        word_dct = {}
        for k in word_list:
            if k not in word_dct:
                word_dct[k] = 1
            else:
                word_dct[k] += 1
        #去掉功能词（即代词、数词、冠词、助动词和情态动词，部分副词、介词、连词和感叹词）
        meaningless_word = {'the','of','a','in','and','to','for','my','','on',
                            'la','with','2','de','i','it','ii'}
        for i in meaningless_word:
            del(word_dct[i])
        #打印出现频率最高的10个词
        newdct = sorted(word_dct.items(), key = lambda d:d[1], reverse = True)
        word_dct_ten=[]
        for i in range(10):
            word_dct_ten.append(newdct[i])
        return(word_dct_ten)

    def question2(self):
        '''
        计算不同职业的评分高低
        难点：表连接和groupby

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
        难点：多个条件group by和可视化

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
        age_group = pd.cut(uid_age_rating_df['age'],bins=self.age_bin)# cut左开右闭，且数据集中没有>100岁的人
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
        不同性别不同年龄段评分高的电影类型
        难点：三张表连接的后续处理（多重索引等）和可视化

        Returns
        -------
        genre_gender_age_mean_rating_df : DataFrame
            所有电影类别在不同性别不同年龄段的评分均值
        genre_gender_age_var_rating_df : DataFrame
            所有电影类别在不同性别不同年龄段的评分方差

        '''
        user_df = pd.DataFrame(self.user_info, columns=['user id', 'age', 'gender'])
        
        rating_df = pd.DataFrame(self.rating_info, columns=['user id', 'movie id', 'rating'])
        
        genre_list = self.genre_info['genre'].tolist() # 获得所有movie类型
        columns_need = ['movie id']
        columns_need.extend(genre_list)
        movie_df = pd.DataFrame(self.movie_info, columns = columns_need)
        
        user_rating_df = pd.merge(user_df, rating_df, on='user id')
        
        user_rating_movie_df = pd.merge(user_rating_df,
                                       movie_df,
                                       how='left',
                                       on='movie id')
        
        # 分组输出
        age_group = pd.cut(user_rating_movie_df['age'],bins=self.age_bin)
        genre_gender_age_mean_rating_df = pd.DataFrame()
        genre_gender_age_var_rating_df = pd.DataFrame()
        for genre in genre_list:
            # 裁切当前DataFrame，提高运行效率
            cur_user_rating_movie_df = user_rating_movie_df[['age','gender',genre,'rating']]
            # 修改列名，方便后续分组，同时让分组聚合结果插入到genre_gender_age_rating_df中具有合理的语义信息
            cur_user_rating_movie_df.rename(columns={genre:'is current genre'},inplace=True)
            # 按照三重索引group by
            cur_genre_gender_age_rating_df = cur_user_rating_movie_df.groupby([age_group, 'gender', 'is current genre'])
            # 拼接所需要的平均值和方差信息
            genre_gender_age_mean_rating_df[genre] = cur_genre_gender_age_rating_df.agg('mean')['rating']
            genre_gender_age_var_rating_df[genre] = cur_genre_gender_age_rating_df.agg('var')['rating']
        
        return genre_gender_age_mean_rating_df, genre_gender_age_var_rating_df
        
        
if __name__ == '__main__':
    info = Information()
    info.load_info()
    # genre_df = info.question1()
    
    occu_rating_mean_df, occu_rating_var_df = info.question2()
    
    age_rating_mean_df, age_rating_var_df = info.question3()
    
    genre_gender_age_mean_rating_df, genre_gender_age_var_rating_df = info.question4()    






