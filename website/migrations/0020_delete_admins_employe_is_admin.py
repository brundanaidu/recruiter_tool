# Generated by Django 5.0.1 on 2024-06-02 10:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0019_remove_employe_id_alter_employe_eid'),
    ]

    operations = [
        migrations.DeleteModel(
            name='admins',
        ),
        migrations.AddField(
            model_name='employe',
            name='is_admin',
            field=models.BooleanField(default=False),
        ),
    ]
