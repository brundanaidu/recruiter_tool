# Generated by Django 5.0.1 on 2024-05-27 10:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0016_employe_admins'),
    ]

    operations = [
        migrations.AlterField(
            model_name='admins',
            name='eid',
            field=models.CharField(max_length=200),
        ),
    ]
