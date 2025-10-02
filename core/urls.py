from django.urls import path
from . import views


urlpatterns = [
    path('', views.subreddit_input_view, name='subreddit_input'),
    path('eda/<str:subreddit>/', views.eda_results_view, name='eda_results')
]
