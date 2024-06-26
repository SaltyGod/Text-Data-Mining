import heapq
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
from statsmodels.stats.outliers_influence import variance_inflation_factor

class RegressionModel:
    def __init__(self, csv_file, num_features,output_file):
        self.csv_file = csv_file
        self.num_features = num_features
        self.df = None
        self.df_test = None
        self.X = None
        self.y = None
        self.output_file = output_file
    
    def load_and_filter_data(self):
        self.df = pd.read_csv(self.csv_file)
        self.df = self.df.loc[self.df['pay_type'].isin([0, 1, 2])]
        self.df_test = self.df[self.df['labels'] == 2].reset_index(drop=True)
        self.df_test = self.df_test[(self.df_test['price'] > 0) & (self.df_test['price'] <= 12000000)]
        print("Number of selected rows: ", len(self.df_test))
    
    def extract_tfidf_features(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df_test['desc_process'].values.astype('U'))
        column_sum = np.sum(tfidf_matrix, axis=0)
        column_sum_2 = column_sum.tolist()
        word_sort = tfidf_vectorizer.get_feature_names_out()
        d = dict(zip(word_sort, column_sum_2[0]))
        sorted_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        top_n = heapq.nlargest(self.num_features, d, key=d.get)
        print(top_n)

        list_sort = []
        for i in range(self.num_features):
            bool_array = np.char.equal(word_sort.tolist(), top_n[i])
            indices = np.where(bool_array)
            a = [i for i, *_ in indices]
            list_sort.append(a[0])
        x = tfidf_matrix[:, list_sort]

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(x.toarray())

        self.df_test['pay_type'] = self.df_test['pay_type'].astype('category')
        dummies = pd.get_dummies(self.df_test['pay_type'], prefix='pay_type')
        dummies = dummies.to_numpy()
        self.X = np.column_stack((X_normalized, dummies[:, 1:]))
        self.y = self.df_test['price']
    
    def forward_selected(self, method='aic'):
        variate = set(range((self.X.shape[1]) - 2))
        selected = [20, 21]
        current_score, best_new_score = np.inf, np.inf
        while variate:
            score_with_variate = []
            for candidate in variate:
                Xc = self.X[:, selected + [candidate]]
                Xc = sm.add_constant(Xc)
                if method == 'aic':
                    score = sm.OLS(self.y, Xc).fit().aic
                else:
                    score = sm.OLS(self.y, Xc).fit().bic
                score_with_variate.append((score, candidate))
            score_with_variate.sort(reverse=False)
            best_new_score, best_candidate = score_with_variate.pop(0)
            if current_score > best_new_score:
                variate.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
            else:
                break

        X_final1 = self.X[:, selected]
        X_final = sm.add_constant(X_final1)
        ols_res1 = sm.OLS(self.y, X_final).fit(cov_type='HC0', use_t=True)
        ols_res2 = sm.OLS(self.y, X_final).fit()

        print(ols_res1.summary())
        print(ols_res2.summary())
        print(selected)

        print("====================多重共线性检验====================")
        vif = [variance_inflation_factor(X_final1, i) for i in range(X_final1.shape[1])]
        print(vif)

        self.save_summary_to_excel(ols_res1.summary().tables[1])
    
    def save_summary_to_excel(self, table):
        df = pd.DataFrame(table.data[1:], columns=table.data[0])
        wb = Workbook()
        ws = wb.active
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
            
        # 检查文件
        output_file = self.output_file
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        wb.save(output_file)


if __name__ == '__main__':
    
    csv_file = './data/selected_data_with_label.csv'
    output_file = './output/reg_model_summary.xlsx'
    num_features = 20
    model = RegressionModel(csv_file, num_features, output_file)
    model.load_and_filter_data()
    model.extract_tfidf_features()
    model.forward_selected(method='aic')
