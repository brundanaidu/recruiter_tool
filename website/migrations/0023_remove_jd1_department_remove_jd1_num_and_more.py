# Generated by Django 5.0.1 on 2024-06-03 06:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0022_remove_employe_groups_remove_employe_is_superuser_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='jd1',
            name='department',
        ),
        migrations.RemoveField(
            model_name='jd1',
            name='num',
        ),
        migrations.AlterField(
            model_name='jd1',
            name='description',
            field=models.FileField(upload_to=''),
        ),
    ]
