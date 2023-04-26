from django.shortcuts import render, redirect
import requests
from bs4 import BeautifulSoup
from django.http import HttpResponse
import kaggle
import os
from .forms import MyModelForm
from django.conf import settings
from django.http import HttpResponse
import pandas as pd
import numpy as np
import csv

# 当前选中的数据集
select_file = ''
# 当前选中数据集的dataframe
select_df = pd.DataFrame()


def base(request):
    return render(request, "template.html")

def child(request):
    return render(request, "child.html")


###################针对table.html##########################
# 文本文件转csv文件
def txt_to_csv(txt_file_path, csv_file_path):
    # 获取TXT文件路径和名称
    txt_file_name = txt_file_path.split('/')[-1]
    csv_file_name = txt_file_name.split('.')[0] + '.csv'
    csv_file_path = txt_file_path.replace(txt_file_name, csv_file_name)

    # 读取TXT文件数据
    with open(txt_file_path, 'r') as txtfile:
        data = txtfile.readlines()

    # 将数据写入CSV文件
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in data:
            # 分割每行数据，并转换为列表形式
            row = line.strip().split('\t')
            writer.writerow(row)
    print(f'{txt_file_name} has been converted to {csv_file_name}')

    return csv_file_path

# 上传文件
def upload_file(request):
    global select_file
    global select_df
    if request.method == 'POST':
        file = request.FILES['file']
        # 更新选中的数据集
        select_file = str(file)
        ##########区分csv文件和excel文件
        # 获取文件扩展名
        file_extension = select_file.split('.')[-1]

        # 如果是CSV文件，则使用read_csv函数读取数据
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension == 'txt':
            file = txt_to_csv(file)
            df = pd.read_csv(file)
        else:
        # 读取上传的数据
            df = pd.read_excel(file)
        print("读取文件成功！")
        # 更新选中数据集的dataframe
        select_df = df
        # 动态获取数据集的键和值
        columns = df.columns.tolist()
        # 只取几列进行显示
        data_head = df[:5]
        values = data_head.values.tolist()

        # 基本统计信息
        # 获取数据集的形状
        shape = df.shape
        # 输出数据集的形状和大小
        data_shape = str(shape[0]) + "行x" + str(shape[1]) + "列"
        # 获取每列的数据类型
        data_types = df.dtypes.tolist()
        # 获取每一列的缺失值数量
        missing_values = df.isnull().sum().tolist()
        # 计算每一列的离群值
        num_outliers = []
        for col in df.columns:
            if df[col].dtype == 'float' or df[col].dtype == 'int':
                # 使用 Z-分数方法计算离群值， 如果某个值的 Z-分数大于 3，则将其视为异常值
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                num_outliers.append(sum(z_scores > 3))
            else:
                num_outliers.append('该数据类型不支持该处理！')

        # 基本数值统计量
        avg_list = []
        for feature in df.columns:
            if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                median = round(df[feature].mean(), 2)
            else:
                median = '该数据类型不支持该处理！'
            avg_list.append(median)
        # 标准差
        std_list = []
        for feature in df.columns:
            if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                median = round(df[feature].std(), 2)
            else:
                median = '该数据类型不支持该处理！'
            std_list.append(median)
        min_list = []
        for feature in df.columns:
            if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                median = round(df[feature].min(), 2)
            else:
                median = '该数据类型不支持该处理！'
            min_list.append(median)
        max_list = []
        for feature in df.columns:
            if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                median = round(df[feature].max(), 2)
            else:
                median = '该数据类型不支持该处理！'
            max_list.append(median)
        median_list = []
        for feature in df.columns:
            if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                median = round(df[feature].median(), 2)
            else:
                median = '该数据类型不支持该处理！'
            median_list.append(median)


        filename = file.name
        filepath = os.path.join(settings.MEDIA_ROOT, filename)
        with open(filepath, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        file_url = '/media/' + filename
        print("fi:", file_url)

        # 将数据传递给模板
        context = {
            'columns': columns,
            'values': values,
            'file_url': file_url,
            'file_title':filename,
            'avg_list':avg_list,
            'std_list':std_list,
            'max_list':max_list,
            'min_list':min_list,
            'median_list':median_list,
            'data_shape':data_shape,
            'data_types':data_types,
            'missing_values':missing_values,
            'num_outliers':num_outliers
        }

        return render(request, 'table.html', context)
    else:
        if select_file != '':
            # 动态获取数据集的键和值
            columns = select_df.columns.tolist()

            df = select_df
            # 只取几列进行显示
            data_head = df[:5]
            values = data_head.values.tolist()

            # 基本统计信息
            # 获取数据集的形状
            shape = df.shape
            # 输出数据集的形状和大小
            data_shape = str(shape[0]) + "行x" + str(shape[1]) + "列"
            # 获取每列的数据类型
            data_types = df.dtypes.tolist()
            # 获取每一列的缺失值数量
            missing_values = df.isnull().sum().tolist()
            # 计算每一列的离群值
            num_outliers = []
            for col in df.columns:
                if df[col].dtype == 'float' or df[col].dtype == 'int':
                    # 使用 Z-分数方法计算离群值， 如果某个值的 Z-分数大于 3，则将其视为异常值
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    num_outliers.append(sum(z_scores > 3))
                else:
                    num_outliers.append('该数据类型不支持该处理！')

            # 基本数值统计量
            avg_list = []
            for feature in df.columns:
                if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                    median = round(df[feature].mean(), 2)
                else:
                    median = '该数据类型不支持该处理！'
                avg_list.append(median)
            # 标准差
            std_list = []
            for feature in df.columns:
                if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                    median = round(df[feature].std(), 2)
                else:
                    median = '该数据类型不支持该处理！'
                std_list.append(median)
            min_list = []
            for feature in df.columns:
                if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                    median = round(df[feature].min(), 2)
                else:
                    median = '该数据类型不支持该处理！'
                min_list.append(median)
            max_list = []
            for feature in df.columns:
                if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                    median = round(df[feature].max(), 2)
                else:
                    median = '该数据类型不支持该处理！'
                max_list.append(median)
            median_list = []
            for feature in df.columns:
                if df[feature].dtype == 'float' or df[feature].dtype == 'int':
                    median = round(df[feature].median(), 2)
                else:
                    median = '该数据类型不支持该处理！'
                median_list.append(median)

            file = select_file
            # filepath = os.path.join(settings.MEDIA_ROOT, file)
            # with open(filepath, 'wb+') as destination:
            #     for chunk in file.chunks():
            #         destination.write(chunk)
            file_url = '/media/' + file
            print("fi:", file_url)

            # 将数据传递给模板
            context = {
                'columns': columns,
                'values': values,
                'file_url': file_url,
                'file_title': file,
                'avg_list': avg_list,
                'std_list': std_list,
                'max_list': max_list,
                'min_list': min_list,
                'median_list': median_list,
                'data_shape': data_shape,
                'data_types': data_types,
                'missing_values': missing_values,
                'num_outliers': num_outliers
            }
            return render(request, 'table.html', context)
        else:
            print("eror!!!!")
        return render(request, 'table.html')


# 下载文件
def download_file(request, filename):
    filepath = os.path.join(settings.MEDIA_ROOT, filename)
    with open(filepath, 'rb') as file:
        response = HttpResponse(file.read(), content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename=%s' % filename
        response['X-Sendfile'] = filepath
        return response


###################针对dashboard.html######################
def line_chart(request):
    print('select_file',select_file)
    df = select_df
    # 动态获取数据集的键和值

    # 提取每个int特征为列表
    all_int_list = []
    all_string_list = []
    columns = []
    all_column = []

    for i in range(0, df.shape[0]):
        all_column.append(i+1)
    for i, col in enumerate(df.columns):
        if df[col].dtype == 'float' or df[col].dtype == 'int':
            list1 = df.iloc[:, i].tolist()
            # 每条折线的数据
            all_int_list.append(list1)
            # 有哪些折线
            columns.append(col)
        else:
            pie_data_index = list(df[col].value_counts().index)
            pie_data = list(df[col].value_counts())
            for i in range(len(pie_data)):
                dic = {}
                dic['name'] = pie_data_index[i]
                dic['value'] = pie_data[i]
                all_string_list.append(dic)

#  散点图
    san_list = []
    for j, k in zip(df[columns[0]], df[columns[1]]):
        san_list.append([j, k])

    context = {
        'all_int_list':all_int_list,
        'all_column':all_column,
        'columns':columns,
        'san_list':san_list
    }
    return render(request, 'dashboard.html', context)


# Create your views here.
def search_datasets(request):
    if request.method == 'POST':
        keyword = request.POST['keyword']
        url = f'https://www.kaggle.com/datasets?search={keyword}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        datasets = soup.find_all('div', {'class': 'dataset-item'})
        dataset_info = []
        for dataset in datasets:
            name = dataset.find('h3').text.strip()
            description = dataset.find('p', {'class': 'dataset-item__description'}).text.strip()
            download_link = dataset.find('a', {'class': 'dataset-item__download-link'})['href']
            dataset_info.append({'name': name, 'description': description, 'download_link': download_link})
            print(name, description, download_link)
        print("datasets", datasets)
        # Write dataset info to a file
        filename = f'{keyword}_datasets.txt'
        with open(filename, 'w') as f:
            for dataset in dataset_info:
                f.write(f'{dataset["name"]}\n{dataset["description"]}\n{dataset["download_link"]}\n\n')

        # Return the file as an HTTP response
        with open(filename, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/plain')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            # os.remove(filename)  # Remove the file from the server
            return response
    else:
        keyword = request.GET.get('keyword')

        # 使用Kaggle API search命令搜索包含指定关键字的数据集
        datasets = kaggle.api.datasets_list(search=keyword)
        print("---"*10)
        # 遍历符合条件的所有数据集，下载到根目录下的'datasets'文件夹里面
        for dataset in datasets:
            if keyword.lower() in dataset.title.lower():
                print(f"Downloading {dataset.ref}...")
                os.system(f"kaggle datasets download {dataset.ref} -p datasets")

        return HttpResponse("Data downloaded successfully!")



def index(request):
    return render(request, "table.html")


def sandian(request):
    if request.method == 'POST':
        print('select_file', select_file)
        df = select_df
        # 动态获取数据集的键和值

        # 提取每个int特征为列表
        all_int_list = []
        all_string_list = []
        columns = []
        all_column = []

        for i in range(0, df.shape[0]):
            all_column.append(i + 1)
        for i, col in enumerate(df.columns):
            if df[col].dtype == 'float' or df[col].dtype == 'int':
                list1 = df.iloc[:, i].tolist()
                # 每条折线的数据
                all_int_list.append(list1)
                # 有哪些折线
                columns.append(col)
            else:
                pie_data_index = list(df[col].value_counts().index)
                pie_data = list(df[col].value_counts())
                for i in range(len(pie_data)):
                    dic = {}
                    dic['name'] = pie_data_index[i]
                    dic['value'] = pie_data[i]
                    all_string_list.append(dic)

        #  散点图
        selected_columns = request.POST.getlist('item')
        san_list = []
        for j, k in zip(select_df[selected_columns[0]], select_df[selected_columns[1]]):
            san_list.append([j, k])

        context = {
            'all_int_list': all_int_list,
            'all_column': all_column,
            'columns': columns,
            'san_list': san_list
        }
        print('san_list', san_list)
        return render(request, 'dashboard.html', context)

