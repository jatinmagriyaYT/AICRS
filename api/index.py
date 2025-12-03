import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')

app = get_wsgi_application()
