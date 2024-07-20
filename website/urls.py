from django.urls import path
from .views import *

urlpatterns=[
    path("",index,name='index'),
    path("home/",home,name='home'),
    path('matcher/',matcher_view,name='matcher'),
    path('login',logins,name='login'),
    # path('register',register,name='register'),
    path('logout',logouts,name='logout'),
    path('add/',addjd,name='addjd'),

    path('delete/<int:id>/',deletejd,name='deletejd'),
    path('update/<int:id>/',updatejd,name='updatejd'),
    path('searchbar/',searchbar,name='searchbar'),
    path('upload_file/',upload_file,name='upload_file'),
    path('delete_profile/<int:id>/',mp,name='deletemp'),

    path('csv/',csv_info,name='csv_info'),

]