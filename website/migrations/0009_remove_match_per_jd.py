# Generated by Django 5.0.1 on 2024-05-03 11:11

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0008_match_per_job_id'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='match_per',
            name='jd',
        ),
    ]
