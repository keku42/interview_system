from django.urls import path    #by 
from . import views

urlpatterns = [
    path("", views.index, name='index'),
    
    path("Round1", views.Round1, name='Round1'),
    path("Round2", views.Round2, name='Round2'),
    path("Round3", views.Round3, name='Round3'),
    path("query", views.query, name='query'),
    
    
]