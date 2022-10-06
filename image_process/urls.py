from django.urls import path

from . import views

app_name = 'detector_app'

urlpatterns = [
    path('',views.index,name='index'),
    path('',views.index,name='getpic')
]