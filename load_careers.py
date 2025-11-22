#!/usr/bin/env python
"""
Script to load career data from CSV into Django database
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

import pandas as pd
from ai_recommender.models import Career

def load_careers():
    """Load career data from CSV into database"""
    csv_path = 'datasets/career_dataset.csv'

    try:
        # Read CSV file as raw text and parse manually
        with open(csv_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Clear existing careers (optional)
        Career.objects.all().delete()

        # Skip header line
        for line in lines[1:]:
            try:
                # Split by comma and handle quoted fields
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue

                # Extract fields (this is a simplified parser)
                title = parts[1].strip()
                description = parts[2].strip()
                required_skills = parts[3].strip()
                education_required = parts[4].strip()

                # Handle salary and growth rate
                try:
                    average_salary = float(parts[5].strip())
                except (ValueError, TypeError):
                    average_salary = 0.0

                try:
                    job_growth_rate = float(parts[6].strip())
                except (ValueError, TypeError):
                    job_growth_rate = 0.0

                work_environment = parts[7].strip()
                related_fields = parts[8].strip() if len(parts) > 8 else ''

                Career.objects.create(
                    title=title,
                    description=description,
                    required_skills=required_skills,
                    education_required=education_required,
                    average_salary=average_salary,
                    job_growth_rate=job_growth_rate,
                    work_environment=work_environment,
                    related_fields=related_fields
                )
                print(f"Loaded career: {title}")
            except Exception as e:
                print(f"Error loading career from line '{line[:50]}...': {e}")
                continue

        print(f"Successfully loaded {len(lines)-1} careers into database")

    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error loading careers: {e}")

if __name__ == '__main__':
    load_careers()