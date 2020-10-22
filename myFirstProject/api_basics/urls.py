
from django.contrib import admin
from django.urls import path
from .views import train_model,scrap,predict,languages,translate


urlpatterns = [

    path('train/', train_model),
    path('scrap/', scrap),
    path('predict/', predict),
    path('languages/',languages ),
    path('translate/',translate)
]
