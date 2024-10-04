import numpy as np
import pandas as pd
import math
import pandas as pd
import matplotlib.pyplot as plt
from  collections import Counter
import random
import seaborn as sns
import gradio as gr
import io
from sklearn.decomposition import PCA
import preprocessing
import attributeAnalyzer
import FrequentItemsets
import StatisticsCOVID19
import Kmeans
import KNN
import DtClassifier
import DBScan
import RandomForestClassifier
import ClassifierMetrics
import ClusteringMetrics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class App:
    def __init__(self):
        self.df1 = pd.read_csv('datasets\Dataset1.csv')
        self.dt1 = np.genfromtxt('datasets\Dataset1.csv', delimiter=',', dtype=float, skip_header=1)
        self.dataset11 = (np.genfromtxt('datasets\Dataset1.csv', delimiter=',', dtype=float, skip_header=1))[:,:-1]
        self.attribute_analyzer = attributeAnalyzer.AttributeAnalyzer(self.dt1, self.df1)
        self.dataFrame2 = pd.read_csv('datasets\Dataset2.csv')
        self.dataFrame2 = self.dataFrame2.replace({pd.NA: np.nan})
        self.dataset2 = self.dataFrame2.to_numpy()
        self.preprocessor2 = preprocessing.Preprocessing(self.dataset2, self.dataFrame2)
        self.dataFrame3 = pd.read_csv('datasets\Dataset3.xlsx - 8.forFMI.csv', delimiter=',', decimal=',')
        self.dataset3 = self.dataFrame3.to_numpy()
        self.FIL = FrequentItemsets.FrequentItemsets()
        self.selected_attribute_dataset3 = 0
        self.create_interface()
        
    def infos_dataset(self, dataFrame):
        num_rows, num_cols = pd.DataFrame(dataFrame).shape
        attr_desc = pd.DataFrame(dataFrame).describe()
        attr_desc.insert(0, 'Stats', attr_desc.index)
        return num_rows, num_cols, attr_desc
    
    def preprocessing_general1(self, manque_meth, aberrante_meth, normalization_meth, vmin, vmax):
        self.preprocessor1 = preprocessing.Preprocessing(self.dt1, self.df1)
        self.preprocessor1.remplacement_manquant_generale(manque_meth)
        self.preprocessor1.remplacement_aberantes_generale(aberrante_meth)
        self.preprocessor1.reduire_row() 
        self.preprocessor1.reduire_dim(0.75)
        self.preprocessor1.normalisation_generale(normalization_meth, int(vmin), int(vmax))
        self.dataset1 = self.preprocessor1.dataset
        self.dataFrame1 = self.preprocessor1.dataFrame
        self.vmin = vmin
        self.vmax = vmax
        self.manque_meth = manque_meth
        self.aberrante_meth = aberrante_meth
        self.normalization_meth = normalization_meth
        return pd.DataFrame(self.dataset1, columns=[col for col in self.dataFrame1.columns.tolist()])
     
    def preprocessing_general2(self, manque_meth, aberrante_meth):
        for row in self.dataset2:
            row[3] = self.preprocessor2.convert_date(row[1], row[3])
            row[4] = self.preprocessor2.convert_date(row[1], row[4])
        self.preprocessor2.remplacement_manquant_generale2(manque_meth)
        self.preprocessor2.remplacement_aberantes_generale2(aberrante_meth)
        self.dataset2 = self.preprocessor2.dataset
        return self.dataset2
    
    def plots(self, df, graph, graph_type1, attribute1, zone2, attribute2, period2, year2, month2, year22, n5, time_period6, attribute6):
        stats = StatisticsCOVID19.StatisticsCOVID19(df)
        if graph == "Total des cas confirmés et tests positifs par zones":
            if graph_type1 == "Bar Chart":
                plot1 = plt.figure()
                stats.plot_total_cases_and_positive_tests(attribute1)
                plot1.savefig("plots\\plot1.png")
                plt.close(plot1)
                plot = ["plots\\plot1.png"]
                return plot
            
            if graph_type1 == "Tree Map":
                plot1 = stats.plot_total_cases_and_positive_tests_treemap(attribute1)
                plot = [plot1]
                return plot
            
        if graph == "Evolution du virus au fil du temps":
            if period2 == "Weekly":
                plot2 = plt.figure()
                stats.weekly_plot(zone2, year2, month2, attribute2)
                plot2.savefig("plots\\plot2_weekly.png")
                plt.close(plot2)
                plot = ["plots\\plot2_weekly.png"]
                return plot
            
            if period2 == "Monthly":
                plot2 = plt.figure()
                stats.monthly_plot(zone2, year22, attribute2)
                plot2.savefig("plots\\plot2_monthly.png")
                plt.close(plot2)
                plot = ["plots\\plot2_monthly.png"]
                return plot
            
            if period2 == "Annual":
                plot2 = plt.figure()
                stats.annual_plot(zone2, attribute2)
                plot2.savefig("plots\\plot2_annual.png")
                plt.close(plot2)
                plot = ["plots\\plot2_annual.png"]
                return plot
            
        if graph == "Total des cas par zone et par année":
            plot3 = plt.figure()
            stats.stacked_bar_plot()
            plot3.savefig("plots\\plot3.png")
            plt.close(plot3)
            plot = ["plots\\plot3.png"]
            return plot
        
        if graph == "Rapport entre la population et le nombre de tests effectués":
            plot4 = plt.figure()
            stats.pop_tests_plot()
            plot4.savefig("plots\\plot4.png")
            plt.close(plot4)
            plot = ["plots\\plot4.png"]
            return plot
        
        if graph == "Top 5 des zones les plus impactées par le coronavirus":
            plot5 = plt.figure()
            stats.plot_top_zones_impacted(n5)
            plot5.savefig("plots\\plot5.png")
            plt.close(plot5)
            plot = ["plots\\plot5.png"]
            return plot
        
        if graph == "Rapport entre les cas confirmés, les tests effectués et les tests positifs a une periode choisie pour chaque zone":
            plot6 = plt.figure()
            stats.plot_time_period_data(time_period6, attribute6)
            plot6.savefig("plots\\plot6.png")
            plt.close(plot6)
            plot = ["plots\\plot6.png"]
            return plot

    def Discretisation(self, method, attribute, K):
        vals=self.dataset3[:,attribute].copy()
        vals.sort()

        if method == "Equal-Width Discretization":
            #Equal-Width Discretization
            nbrelmt=math.ceil(self.dataset3[:,attribute].shape[0]/K)
            
            for val in range(0,self.dataset3[:,attribute].shape[0]):  
                for i in range(0,vals.shape[0],nbrelmt):
                    if(vals[i]>self.dataset3[val,attribute]):
                        sup=i
                        break
                self.dataset3[val,attribute]=np.mean(vals[sup-nbrelmt:sup])       
        else:
            #Equal-Frequency Discretization
            largeur= (self.dataset3[:,attribute].max() - self.dataset3[:,attribute].min())/math.ceil(K)
            
            dic={}
            bornesup= self.dataset3[:,attribute].min()+largeur
            for val in vals:
                if val>=bornesup and bornesup<self.dataset3[:,attribute].max():
                    bornesup+=largeur

                if bornesup in dic:   
                    dic[bornesup].append(val)
                else:
                    dic[bornesup]=[val]

            for i in range(0,self.dataset3[:,attribute].shape[0]):
                for item in dic.items():
                    if (self.dataset3[i,attribute]>=item[0]-largeur and self.dataset3[i,attribute]<item[0]):
                        self.dataset3[i,attribute]=np.mean(item[1])
                        break

    def discretization_plot(self, method, attribute, K):
        plot = plt.figure()
        #Before Discretization
        plt.hist(self.dataset3[:,attribute], bins=K, edgecolor='black')
        
        #Discretization
        self.Discretisation(method, attribute, K)
        self.transactions_table = np.array([np.sort(list(four)) for four in zip(self.dataset3[:, attribute].astype(str), self.dataset3[:,3], self.dataset3[:,4], self.dataset3[:,5])])

        #After Discretization
        attribute_label = self.dataFrame3.columns[attribute]
        freq=Counter(self.dataset3[:, attribute])
        x_values = np.sort(list(freq.keys()))
        y_values = [freq[x] for x in x_values]
        
        plt.plot(x_values, y_values)
        plt.xlabel(f'{attribute_label}')
        plt.ylabel('Frequency')
        plt.title(f'Discretized {attribute_label}')
        plot.savefig("plots\discretization_plot.png")
        plt.close(plot)
        plot = ["plots\\discretization_plot.png"]
        return self.dataset3, plot, self.transactions_table

    def FIL_general(self, transactions_table, supp_min, conf_min, metric):
        self.transactions_table = pd.DataFrame(transactions_table).to_numpy()
        self.L = self.FIL.appriori(supp_min, self.transactions_table)
        self.association_rules = self.FIL.regles_frequente(self.L, conf_min, metric)
        self.L = [(k, v) for dictionary in self.L for k, v in dictionary.items()]

        return self.association_rules, pd.DataFrame(self.L, columns=['Frequent Itemset', 'Support'])

    def experimentation_plots(self, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound):
        
        plot1 = self.FIL.rules_nbr_plot(self.transactions_table, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound)

        plot2 = plt.figure()
        self.FIL.freq_items_nbr_plot(self.transactions_table, supp_lower_bound, supp_upper_bound)
        plot2.savefig("plots\\freq_items_nbr_plot.png")
        plt.close(plot2)

        plot3 = plt.figure()
        self.FIL.time_exec_plot(self.transactions_table, supp_lower_bound, supp_upper_bound)
        plot3.savefig("plots\\time_exec_plot.png")
        plt.close(plot3)

        plot4 = self.FIL.memory_alloc_plot(self.transactions_table, supp_lower_bound, supp_upper_bound, conf_lower_bound, conf_upper_bound)

        plots = [plot1, "plots\\freq_items_nbr_plot.png", "plots\\time_exec_plot.png", plot4]
        
        return plots

    def recommendation(self, a1, a2, a3, method):
        instance = []
        if a1 != "":
            instance.append(a1)
        if a2 != "":
            instance.append(a2)
        if a3 != "":
            instance.append(a3)

        r_filtered=[]
        for index, row in self.association_rules.iterrows():
            if method == "Strict":
                if row[0]==tuple(sorted(instance)):
                    r_filtered.append(row)
            else:
                if set(list(row[0])).issubset(set(instance)):
                    r_filtered.append(row)

        if r_filtered:
            max_consequent_length = max(len(r["consequent"]) for r in r_filtered)
        else:
            max_consequent_length = 0

        consequent_columns = [f"Consequent {i+1}" for i in range(max_consequent_length)]

        return pd.DataFrame([r["consequent"] for r in r_filtered], columns=consequent_columns)

    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        data = list(zip(X, y))
        random.shuffle(data)

        split_index = int(len(data) * (1 - test_size))

        train_data = data[:split_index]
        test_data = data[split_index:]

        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)

        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        return X_train, X_test, y_train, y_test

    def metric_value(self, metric, minkowski_param):
        if metric == 'Euclidean':
            metric = 2
        if metric == 'Manhattan':
            metric = 1
        if metric == 'Minkowski':
            metric = minkowski_param
        if metric == 'Cosine':
            metric = 0
        return metric 
    
    def confusion_matrix_plot(self, m, model):
        fig, ax = plt.subplots()  
        sns.heatmap(m, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix {model}")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)

        with open(f"plots\\confusion_matrix_{model}.png", 'wb') as f:
            f.write(buffer.getvalue())

        return f"plots\\confusion_matrix_{model}.png"

    def clustering_plots(self, res0, res):
        plt.scatter([r[0] for r in res0], [r[1] for r in res0], c=res, cmap='cividis', marker='H', edgecolors='k')
        plt.title('Clustering with PCA')
        plt.xlabel('component 1 of PCA')
        plt.ylabel('componant 2 of PCA')

    def classification(self, model, metric, minkowski_param, knn_param, min_samples_split_DT, max_depth_DT, info_gain_metric_DT, min_samples_split_RF, max_depth_RF, nbr_trees_RF, nbr_features_RF, info_gain_metric_RF, instance):
        X_train, X_test, Y_train, Y_test =  self.train_test_split(self.dataset1[:, :-1], self.dataset1[:, -1], test_size=0.2, random_state=42)
        y_instance = None
        instance = np.array([[float(item) for item in inner_list] for inner_list in instance])
        dt1 = np.vstack([self.dataset11, instance])
        dataset_instance = dt1

        metric = int(self.metric_value(metric, int(minkowski_param)))

        self.preprocessor_instance = preprocessing.Preprocessing(dataset_instance, pd.DataFrame(dataset_instance))
        self.preprocessor_instance.remplacement_manquant_generale(self.manque_meth)
        self.preprocessor_instance.remplacement_aberantes_generale(self.aberrante_meth)
        self.preprocessor_instance.reduire_row() 
        self.preprocessor_instance.reduire_dim(0.75)
        self.preprocessor_instance.normalisation_generale(self.normalization_meth, int(self.vmin), int(self.vmax)) 
        
        dataset = self.preprocessor_instance.dataset
        instance = dataset[-1]

        if model == 'KNN':
            KNNClassifier = KNN.KNN(int(knn_param), metric)
            KNNClassifier.fit(X_train, Y_train) 
            y_pred=[]
            for i in X_test:
                y_pred.append(KNNClassifier._predict(i))
            y_instance=KNNClassifier._predict(instance)

        if model == 'Decision Trees':
            DTClassifier = DtClassifier.DtClassifier(min_samples_split=int(min_samples_split_DT), max_depth=int(max_depth_DT), info_gain_method=info_gain_metric_DT)
            DTClassifier.fit(X_train, Y_train)
            y_pred = DTClassifier.predict(X_test)
            y_instance=DTClassifier.predict(np.array([instance]))

        if model == 'Random Forest':
            random_forest = RandomForestClassifier.RandomForestClassifier(n_trees=int(nbr_trees_RF), max_depth=int(max_depth_RF), min_samples_split=int(min_samples_split_RF), n_features=int(nbr_features_RF), info_gain_method=info_gain_metric_RF)
            random_forest.fit(X_train, Y_train)
            y_pred = random_forest.predict(X_test) 
            y_instance=random_forest.predict(np.array([instance]))

        self.ClassifierMetrics = ClassifierMetrics.ClassifierMetrics(Y_test, y_pred)
        
        confusion_matrix = self.ClassifierMetrics.confusion_matrix(Y_test, y_pred)
        TP, FN, FP, TN = self.ClassifierMetrics.Values(confusion_matrix)
        accuracy = self.ClassifierMetrics.accuracy_score(confusion_matrix)
        recall = self.ClassifierMetrics.recall_score(TP, FN).tolist()
        precision = self.ClassifierMetrics.precision_score(TP, FP).tolist()
        FP_rate = self.ClassifierMetrics.FP_rate(FP, TN).tolist()
        specificity = self.ClassifierMetrics.specificity_score(TN, FP).tolist()
        f1_score = self.ClassifierMetrics.f1_score(TP, FP, FN).tolist()
            
        recall.append(np.mean(recall))
        precision.append(np.mean(precision))
        FP_rate.append(np.mean(FP_rate))
        specificity.append(np.mean(specificity))
        f1_score.append(np.mean(f1_score))

        data = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "FP Rate": FP_rate,
            "Specificity": specificity,
            "F1 Score": f1_score
        }

        classification_report = pd.DataFrame(data)

        class_labels = [f"Class_{i}" for i in range(len(recall)-1)]
        class_labels.append("Global")
        classification_report["Class"] = class_labels

        classification_report.set_index("Class", inplace=True)

        conf_matrix_plot = self.confusion_matrix_plot(confusion_matrix, model)

        return classification_report, y_instance, [conf_matrix_plot]

    def clustering(self, model, metric, minkowski_param, n_cluster_km, centroid_select_method_km, max_iterations_km, min_samples_db, radius_db, pca_clust, instance):
        metric = self.metric_value(metric, int(minkowski_param))
        dataset = self.dataset1[:, :-1]
        prediction = None
        if pca_clust == "Yes":
            pca = PCA(n_components=2)
            dataset = pca.fit_transform(dataset)
        else:
            pca = None
            plot = None
        if model == 'K-Means':
            instance = np.array([[float(item) for item in inner_list] for inner_list in instance])
            dt1 = np.vstack([self.dataset11, instance])

            self.preprocessor_instance = preprocessing.Preprocessing(dt1, pd.DataFrame(dt1))
            self.preprocessor_instance.remplacement_manquant_generale(self.manque_meth)
            self.preprocessor_instance.remplacement_aberantes_generale(self.aberrante_meth)
            self.preprocessor_instance.reduire_row() 
            self.preprocessor_instance.reduire_dim(0.75)
            self.preprocessor_instance.normalisation_generale(self.normalization_meth, int(self.vmin), int(self.vmax)) 
            dt1 = self.preprocessor_instance.dataset
            if pca_clust == 'Yes':
                dt1 = pca.fit_transform(dt1)
            instance = dt1[-1]

            kmeansClustering = Kmeans.K_MEANS(k=int(n_cluster_km),methode_d=metric,methode_c=centroid_select_method_km,max_iterations=max_iterations_km, dataset=dataset)#k=2, pca=2, methode_d2 methode_c 1 3000 800
            kmeansClustering.fit(dataset)
            res=kmeansClustering._cluster()
            res0 = res[:, :-1]
            res = res[:, -1]
            km_labeled_dataset = np.concatenate((self.dataset1[:, :-1], res.reshape(-1, 1)), axis=1)
            prediction=kmeansClustering._prediction(instance.tolist())[0]
            labeled_dataset = pd.DataFrame(km_labeled_dataset, columns=[f"feature_{i+1}" for i in range((km_labeled_dataset.shape[1])-1)] + ["cluster_label"])

        else:
            DBSCANClustering=DBScan.DB_Scan(radius_db, min_samples_db, methode_d=metric, dataset=dataset)# 1.2 5 0.45  1/0/1
            res0 = np.array(DBSCANClustering[0])
            res = np.array(DBSCANClustering[1])

            DBSCAN_labeled_dataset = np.concatenate((self.dataset1[:, :-1], res.reshape(-1, 1)), axis=1)
            labeled_dataset = pd.DataFrame(DBSCAN_labeled_dataset, columns=[f"feature_{i+1}" for i in range((DBSCAN_labeled_dataset.shape[1])-1)] + ["cluster_label"])
            
        self.ClusteringMetrics = ClusteringMetrics.ClusteringMetrics(res0, res)
        silhouette_score, intra_distance, inter_distance = self.ClusteringMetrics.silhouette_score(res0, res, metric)


        if pca_clust=="Yes":
            plot1 = plt.figure()
            self.clustering_plots(res0, res)
            plot1.savefig("plots\\clustering_PCA.png")
            plt.close(plot1)
            plot = ["plots\\clustering_PCA.png"]

        data = {
            "Silhouette": [silhouette_score],
            "Intra Cluster Distance": [intra_distance],
            "Inter Cluster Distance": [inter_distance]
        }

        clustering_report = pd.DataFrame(data)

        return labeled_dataset, clustering_report, prediction, plot

    def create_interface(self):
        with gr.Blocks() as demo:
            with gr.Tab("Agriculture"):
                with gr.Tab("Dataset Visualization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset1 Analysis Dashboard""")
                        
                        with gr.Row():
                            inputs = [gr.Dataframe(label="Dataset1", value=self.df1)]
                           
                            with gr.Column():
                                outputs = [gr.Textbox(label="Number of Rows"), gr.Textbox(label="Number of Columns"), gr.Dataframe(label="Attributes description")]
                        
                        btn = gr.Button("Submit")
                        btn.click(fn=self.infos_dataset, inputs=inputs, outputs=outputs)

                with gr.Tab("Attributes"):
                    with gr.Column():
                        gr.Markdown("""# Attributes Analysis""")
                        
                        with gr.Row():
                            inputs = [gr.Dropdown([(f"{att}", i) for i, att in enumerate(self.df1.columns.tolist())], multiselect=False, label="Attributes", info="Select an attribute : "), 
                                      gr.Radio(["With Outliers", "Without Outliers"], label="Box Plot Parameters"), 
                                      gr.Dropdown([(f"{att}", i) for i, att in enumerate(self.df1.columns.tolist())], multiselect=False, label="Scatter Plot Parameters", info="Select a second attribute for the scatter plot : ")]
                        
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    trends = [gr.Textbox(label="Mean"), gr.Textbox(label="Median"), gr.Textbox(label="Mode")]
                                with gr.Row():
                                    quartiles = [gr.Textbox(label="Q0"), gr.Textbox(label="Q1"), gr.Textbox(label="Q2"), gr.Textbox(label="Q3"), gr.Textbox(label="Q4")]
                                with gr.Row():
                                    deviation = [gr.Textbox(label="Standard Deviation")]

                            with gr.Column():
                                gallery = [gr.Gallery(label="Attribute Visualisation", columns=(1,2))]
                            
                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.attribute_analyzer.attribute_infos, inputs=inputs, outputs=trends+quartiles+deviation+gallery)
            
                with gr.Tab("Preprocessing"):
                    with gr.Column():
                        gr.Markdown("""# Preprocessing of Dataset1""")

                        with gr.Row():
                            inputs = [gr.Dropdown(["Mode", "Mean"], multiselect=False, label="Missing Values", info="Select a method to handle the missing values in the dataset :"), 
                                gr.Dropdown(["Linear Regression", "Discritisation", "Winorisation"], multiselect=False, label="Outliers", info="Select a method to handle the outliers in the dataset :"), 
                                gr.Dropdown(["Vmin-Vmax", "Z-Score"], multiselect=False, label="Normalization", info="Select a method to normalize the dataset :"),
                                gr.Textbox(label="Vmin", visible=True, interactive=True, value=0),
                                gr.Textbox(label="Vmax", visible=True, interactive=True, value=0)]
                            
                        with gr.Row():
                            outputs = [gr.Dataframe(label="Dataset1 preprocessed")]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            dataset1_reprocessing_btn = gr.Button("Submit")
                            dataset1_reprocessing_btn.click(fn=self.preprocessing_general1, inputs=inputs, outputs=outputs)

                with gr.Tab("Classification"):
                    with gr.Tab("Dataset 1 Classification"):
                        with gr.Column():
                            
                            with gr.Row():
                                model = [gr.Dropdown(["KNN", "Decision Trees", "Random Forest"], multiselect=False, label="Model", info="Select a classification model :")] 
                                metric = [gr.Dropdown(["Euclidean", "Manhattan", "Minkowski", "Cosine"], multiselect=False, label="Metric", info="Select a metric :")]
                                minkowski_param = gr.Number(visible=False, value=3, minimum=1, maximum=10, step=1, label="Minkowski P parameter")

                                with gr.Row(visible=False) as knn_param_row:
                                    knn_param = [gr.Number(value=3, minimum=2, maximum=5, step=1, label="K")]

                                with gr.Row(visible=False) as DT_param_row:
                                    DT_param = [gr.Number(value=2, minimum=1, maximum=10, step=1, label="Minimum samples split"), 
                                           gr.Number(value=5, minimum=2, maximum=10, step=1, label="Maximum depth"), 
                                           gr.Dropdown(["Gini", "Entropy"], multiselect=False, label="Information gain metric", info="Select an information gain metric :")]
                                
                                with gr.Row(visible=False) as RF_param_row:
                                    RF_param = [gr.Number(value=2, minimum=1, maximum=10, step=1, label="Minimum samples split"), 
                                           gr.Number(value=6, minimum=2, maximum=12, step=1, label="Maximum depth"),
                                           gr.Number(value=30, minimum=10, maximum=500, step=10, label="Number of trees"),
                                           gr.Number(value=6, minimum=2, maximum=12, step=1, label="Number of features"),
                                           gr.Dropdown(["Gini", "Entropy"], multiselect=False, label="Information gain metric", info="Select an information gain metric :")]
                            with gr.Column():
                                gr.Markdown(""" Insert an instance to predict its class :""")
                                instance = [gr.List(col_count=13)]

                            def update_visibility_model_param(selected_model):
                                    if selected_model == "KNN":
                                        return {knn_param_row: gr.Row(visible=True),
                                                DT_param_row: gr.Row(visible=False),
                                                RF_param_row: gr.Row(visible=False)}
                                    if selected_model == "Decision Trees":
                                        return {knn_param_row: gr.Row(visible=False),
                                                DT_param_row: gr.Row(visible=True),
                                                RF_param_row: gr.Row(visible=False)}
                                    else:
                                        return {knn_param_row: gr.Row(visible=False),
                                                DT_param_row: gr.Row(visible=False),
                                                RF_param_row: gr.Row(visible=True)}
                            model[0].change(update_visibility_model_param, inputs=model[0], outputs=[knn_param_row, DT_param_row, RF_param_row])

                            def update_visibility_minkowski_param(selected_metric):
                                if selected_metric == "Minkowski":
                                    return {minkowski_param: gr.Number(visible=True)}
                                else:
                                    return {minkowski_param: gr.Number(visible=False)}
                            metric[0].change(update_visibility_minkowski_param, inputs=metric[0], outputs=minkowski_param)
                           
                            with gr.Row():
                                with gr.Column():
                                    instance_class = [gr.Textbox(label="Instance Class")]
                                    classification_report = [gr.Dataframe(label="Metrics")]
                                    
                                with gr.Column():
                                    classification_gallery = [gr.Gallery(label="Graphs", columns=(1,2))]
                            
                            btn = gr.Button("Train & Test")
                            btn.click(fn=self.classification, inputs=model+metric+[minkowski_param]+knn_param+DT_param+RF_param+instance, outputs=classification_report+instance_class+classification_gallery)

                with gr.Tab("Clustering"):
                    with gr.Tab("Dataset 1 Clustering"):
                        with gr.Column():
                            
                            with gr.Row():
                                model_clustering = [gr.Dropdown(["DBSCAN", "K-Means"], multiselect=False, label="Model", info="Select a clustering model :")] 
                                metric_clustering = [gr.Dropdown(["Euclidean", "Manhattan", "Minkowski", "Cosine"], multiselect=False, label="Metric", info="Select a metric :")]
                                minkowski_param_clustering = gr.Number(visible=False, value=3, minimum=1, maximum=10, step=1, label="Minkowski P parameter")

                                with gr.Row(visible=False) as KMeans_param_row:
                                    KMeans_param = [gr.Number(value=3, minimum=2, maximum=5, step=1, label="K (Number of clusters)"),
                                                     gr.Dropdown(["Random", "Better picking"], multiselect=False, label="Centroids Selection Methods", info="Choose a method to select the centroids :"),
                                                     gr.Number(value=3000, minimum=100, maximum=10000, step=100, label="Maximum Iterations")]

                                with gr.Row(visible=False) as DBSCAN_param_row:
                                    DBSCAN_param = [gr.Number(value=5, minimum=5, maximum=50, step=5, label="Minimum samples"), 
                                           gr.Number(value=1.2, minimum=0.1, maximum=3.0, step=0.1, label="Radius")]
                                
                                pca = [gr.Radio(['Yes', 'No'], label="PCA")]

                            with gr.Row(visible=False) as clustering_instance_row:
                                clustering_instance = [gr.List(col_count=13, label="Insert an instance to predict its cluster :")]

                            def update_visibility_model_param(selected_model):
                                    if selected_model == "DBSCAN":
                                        return {DBSCAN_param_row: gr.Row(visible=True),
                                                KMeans_param_row: gr.Row(visible=False), 
                                                clustering_instance_row: gr.Row(visible=False), 
                                                instance_cluster_row : gr.Row(visible=False)}
                                    else:
                                        return {DBSCAN_param_row: gr.Row(visible=False),
                                                KMeans_param_row: gr.Row(visible=True), 
                                                clustering_instance_row: gr.Row(visible=True), 
                                                instance_cluster_row : gr.Row(visible=True)}
                                    
                            metric_clustering[0].change(update_visibility_minkowski_param, inputs=metric_clustering[0], outputs=minkowski_param_clustering)
                           
                            with gr.Row():
                                with gr.Column():
                                    labeled_dataset = [gr.DataFrame(label="Labeled Dataset")]
                                    with gr.Row(visible=False) as instance_cluster_row:
                                        instance_cluster = [gr.Textbox(label="Instance Cluster")]
                                    clustering_report = [gr.Dataframe(label="Metrics")]
                                    
                                with gr.Column():
                                    clustering_gallery = [gr.Gallery(label="Graphs", columns=(1,2))]

                            model_clustering[0].change(update_visibility_model_param, inputs=model_clustering[0], outputs=[DBSCAN_param_row, KMeans_param_row, clustering_instance_row, instance_cluster_row])
                            
                            btn = gr.Button("Train")
                            btn.click(fn=self.clustering, inputs=model_clustering+metric_clustering+[minkowski_param_clustering]+KMeans_param+DBSCAN_param+pca+clustering_instance, outputs=labeled_dataset+clustering_report+instance_cluster+clustering_gallery)
               

            with gr.Tab("COVID-19"):
                with gr.Tab("Dataset Visualization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset2 Analysis Dashboard""")
                        
                        with gr.Row():
                            inputs = [gr.Dataframe(label="Dataset2", value=self.dataFrame2)]
                           
                            with gr.Column():
                                outputs = [gr.Textbox(label="Number of Rows"), gr.Textbox(label="Number of Columns"), gr.Dataframe(label="Attributes description")]
                        
                        btn = gr.Button("Submit")
                        btn.click(fn=self.infos_dataset, inputs=inputs, outputs=outputs)

                with gr.Tab("Preprocessing"):
                    with gr.Column():
                        gr.Markdown("""# Preprocessing of Dataset2""")

                        with gr.Row():
                            inputs = [gr.Dropdown(["Mode", "Mean"], multiselect=False, label="Missing Values", info="Select a method to handle the missing values in the dataset :"), 
                                gr.Dropdown(["Linear Regression", "Discritisation"], multiselect=False, label="Outliers", info="Select a method to handle the outliers in the dataset :")]
                            
                        with gr.Row():
                            outputs_preprocess = gr.Dataframe(label="Dataset2 preprocessed", headers=self.dataFrame2.columns.tolist())

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.preprocessing_general2, inputs=inputs, outputs=outputs_preprocess)

                with gr.Tab("Statistics"):
                    with gr.Column():
                        with gr.Row():
                            graph = gr.Dropdown(["Total des cas confirmés et tests positifs par zones", "Evolution du virus au fil du temps", "Total des cas par zone et par année",
                                                 "Rapport entre la population et le nombre de tests effectués", 
                                                 "Top 5 des zones les plus impactées par le coronavirus",
                                                 "Rapport entre les cas confirmés, les tests effectués et les tests positifs a une periode choisie pour chaque zone"], multiselect=False, label="Graphs", info="Select a graph to plot :")
                            with gr.Row(visible=False) as row:
                                plot1_param = [gr.Radio(["Tree Map", "Bar Chart"]),
                                               gr.Dropdown(['case count', 'positive tests'], multiselect=False, label="Atribute", info="Choose an attribute to plot :")]

                            with gr.Row(visible=False) as row2:
                                plot2_param = [gr.Dropdown(self.dataFrame2['zcta'].unique().tolist(), multiselect=False, label="Zone", info="Select a zone to plot :"), 
                                               gr.Dropdown(["case count", "test count", "positive tests"], multiselect=False, label="Attribute", info="Select an attribute to plot :"), 
                                               gr.Dropdown(["Weekly", "Monthly", "Annual"], multiselect=False, label="Period", info="Select a period to plot :")]
                                with gr.Row(visible=False) as row2_w:
                                    weekly_param = [gr.Dropdown([2019, 2020, 2021, 2022], multiselect=False, label="Year", info="Choose a year:"),
                                                    gr.Dropdown([i+1 for i in range(12)], multiselect=False, label="Month", info="Choose a month:")]
                                with gr.Row(visible=False) as row2_m:
                                    monthly_param = [gr.Dropdown([2019, 2020, 2021, 2022], multiselect=False, label="Year", info="Choose a year:")]
                                
                            with gr.Row(visible=False) as row5:
                                plot5_param = [gr.Slider(1, 7, value=5, label="Number of zones", info="Choose between 1 and 7 zones", step=1)]

                            with gr.Row(visible=False) as row6:
                                plot6_param = [gr.Dropdown(np.unique(self.dataset2[:, 1]).tolist(), multiselect=False, label="time period", info="Select a time period to plot :"),
                                    gr.Dropdown(["case count", "test count", "positive tests"], multiselect=False, label="Attribute", info="Select an attribute to plot :")]


                                def update_visibility(selected_graph):
                                    if selected_graph == "Total des cas confirmés et tests positifs par zones":
                                        return {row: gr.Row(visible=True),
                                                row2: gr.Row(visible=False), 
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Evolution du virus au fil du temps":
                                        return {row2: gr.Row(visible=True),
                                                row: gr.Row(visible=False), 
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Total des cas par zone et par année" or selected_graph == "Rapport entre la population et le nombre de tests effectués":
                                        return {row2: gr.Row(visible=False),
                                                row: gr.Row(visible=False),
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Top 5 des zones les plus impactées par le coronavirus":
                                        return {row2: gr.Row(visible=False),
                                                row: gr.Row(visible=False),
                                                row5: gr.Row(visible=True), 
                                                row6: gr.Row(visible=False)}
                                    if selected_graph == "Rapport entre les cas confirmés, les tests effectués et les tests positifs a une periode choisie pour chaque zone":
                                        return {row2: gr.Row(visible=False),
                                                row: gr.Row(visible=False),
                                                row5: gr.Row(visible=False), 
                                                row6: gr.Row(visible=True)}
                                    
                                def update_visibility_param(selected_period):
                                    if selected_period == "Weekly":
                                        return {row2_w: gr.Row(visible=True),
                                                row2_m: gr.Row(visible=False)}
                                    if selected_period == "Monthly":
                                        return {row2_w: gr.Row(visible=False),
                                                row2_m: gr.Row(visible=True)}
                                    else:
                                        return {row2_w: gr.Row(visible=False),
                                                row2_m: gr.Row(visible=False)}
                                    
                                graph.change(update_visibility, inputs=graph, outputs=[row, row2, row5, row6])
                                plot2_param[2].change(update_visibility_param, inputs=plot2_param[2], outputs=[row2_w, row2_m])
                        

                        with gr.Row():
                            outputs = [gr.Gallery(label="Graphs", columns=(1,2))]
                        
                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.plots, inputs=[outputs_preprocess]+[graph]+plot1_param+plot2_param+weekly_param+monthly_param+plot5_param+plot6_param, outputs=outputs)
            
            with gr.Tab("Frequent Itemset Learning"):
                with gr.Tab("Dataset Visualization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Analysis Dashboard""")
                        
                        with gr.Row():
                            inputs = [gr.Dataframe(label="Dataset3", value=self.dataFrame3)]
                           
                            with gr.Column():
                                outputs = [gr.Textbox(label="Number of Rows"), gr.Textbox(label="Number of Columns"), gr.Dataframe(label="Attributes description")]
                        
                        btn = gr.Button("Submit")
                        btn.click(fn=self.infos_dataset, inputs=inputs, outputs=outputs)

                with gr.Tab("Discretization"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Discretization""")
                        
                        with gr.Row():
                            method = [gr.Dropdown(["Equal-Width Discretization", "Equal-Frequency Discretization"], multiselect=False, label="Method", info="Select a method of Discretization :")]
                            attribute3 = gr.Dropdown([(f"{att}", i) for i, att in enumerate(self.dataFrame3.columns.tolist()) if i < 3], value=4, multiselect=False, label="Attributes", info="Select an attribute to discretize :")
                            bins_nbr = gr.Slider(2, 8, step=1, visible=False, label="Bins Number", info="Choose the number of bins :", value=8)
                           
                            def update_visibility_discretization(select_attribute):
                                    self.selected_attribute_dataset3 = select_attribute
                                    if select_attribute == 0:
                                        return {bins_nbr : gr.Slider(maximum=len(np.unique(self.dataset3[:, select_attribute].tolist())), visible=True)}
                                    if select_attribute == 1:
                                        return {bins_nbr : gr.Slider(maximum=len(np.unique(self.dataset3[:, select_attribute].tolist())), visible=True)}
                                    else:
                                        return {bins_nbr : gr.Slider(maximum=len(np.unique(self.dataset3[:, select_attribute].tolist())), visible=True)}
                                    
                            attribute3.change(update_visibility_discretization, inputs=[attribute3], outputs=bins_nbr)
                        with gr.Row():
                            output_dataset3 = [gr.Dataframe(label="Dataset3 after Discretization", headers=self.dataFrame3.columns.tolist())]
                            discretization_gallery = [gr.Gallery(label="Graphs", columns=(1,2))]

                        with gr.Row():
                            gr.ClearButton(method+[attribute3]+[bins_nbr])
                            btn_discr = gr.Button("Submit")
                            
                with gr.Tab("Frequent Itemsets and Association Rules"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Frequent Itemsets and Association Rules""")
                        
                        with gr.Row():
                            transactions_table = gr.Dataframe(label="Transactions Table", headers=["Chosen Attribute", "Soil", "Crop", "Fertilizer"])
                            with gr.Column():    
                                inputs = [gr.Number(label="Minimal Support", value=0.01, step=0.01, minimum=0.01, maximum=1),
                                        gr.Number(label="Minimal Confidence", value=0.1, step=0.01, minimum=0.01, maximum=1),
                                        gr.Dropdown([(f"{m}", i) for i, m in enumerate(["Confidence", "Cosine", "Lift", "Jaccard", "Kulczynski"])], multiselect=False, label="Metric", info="Select a metric for association rules :")]
                            
                        with gr.Row():
                            freq_item = [gr.Dataframe(label="Frequent Itemsets")]
                            rules = [gr.Dataframe(label="Association Rules")]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.FIL_general, inputs=[transactions_table]+inputs, outputs=rules+freq_item)
                            btn_discr.click(fn=self.discretization_plot, inputs=method+[attribute3]+[bins_nbr], outputs=output_dataset3+discretization_gallery+[transactions_table])

                with gr.Tab("Experimentations"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Experimentations""")
                        
                        with gr.Row():  
                            inputs_exp = [gr.Number(label="Lower Bound of Minimal Support", minimum=0.0, maximum=0.3, value=0.01, step=0.01),
                                    gr.Number(label="Upper Bound of Minimal Support", minimum=0.0, maximum=0.3, value=0.2, step=0.01),
                                    gr.Number(label="Lower Bound of Minimal Confidence", minimum=0.0, maximum=0.3, value=0.01, step=0.01),
                                    gr.Number(label="Upper Bound of Minimal Confidence", minimum=0.0, maximum=0.3, value=0.2, step=0.01)]
                        with gr.Row():
                            experimentation_gallery = [gr.Gallery(label="Graphs", columns=(1,2))]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.experimentation_plots, inputs=inputs_exp, outputs=experimentation_gallery)
                
                with gr.Tab("Recommender"):
                    with gr.Column():
                        gr.Markdown("""# Dataset3 Recommender""")
                        gr.Markdown(""" Insert new row :""")
                        with gr.Row():  
                            inputs = [gr.Textbox(label="Antecedent 1", value="Urea"),
                                gr.Textbox(label="Antecedent 2", value="29.283076923076926"),
                                gr.Textbox(label="Antecedent 3", value="Coconut"), 
                                gr.Radio(["Strict", "Not Strict"], label="Method")]
                        with gr.Row():
                            rec = [gr.Dataframe(label="Recommendation Table")]

                        with gr.Row():
                            gr.ClearButton(inputs)
                            btn = gr.Button("Submit")
                            btn.click(fn=self.recommendation, inputs=inputs, outputs=rec)


        self.demo_interface = demo

    def launch(self):
        self.demo_interface.launch()

app = App()
app.launch()