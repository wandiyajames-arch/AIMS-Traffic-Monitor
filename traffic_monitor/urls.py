"""
URL configuration for traffic_monitor project.
"""

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

# Import the views from your dashboard app
from dashboard import views

urlpatterns = [
    path("admin/", admin.site.urls),
    # Point the root URL to your index view (the upload form)
    path("", views.index, name="index"),
    # Point the video_feed URL to your streaming view
    path("video_feed/", views.video_feed, name="video_feed"),
    # Point the download_logs URL to allow users to download the CSV
    path("download_logs/", views.download_logs, name="download_logs"),
]

# This allows Django to serve the uploaded video files locally during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)