"""draggable URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path, re_path, include
from app01 import views
from draggable.settings import MEDIA_ROOT
from django.views.static import serve


urlpatterns = [
    path('admin/', admin.site.urls),
    re_path('^$', views.index),
    path('search/', views.search_datasets, name='search_datasets'),
    path('upload/', views.upload_file, name='upload_file'),
    path('media/<str:filename>/', views.download_file, name='download_file'),
    path('line_chart/', views.line_chart, name = 'line_chart'),
    path('sandian/',views.sandian, name = 'sandian'),
    path('child/', views.child, name='child'),
    path('upload_process/', views.upload_process, name='upload_process'),
    path('data_analytics/', views.data_analytics, name='data_analytics'),



]
