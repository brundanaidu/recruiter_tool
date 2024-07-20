from django.db import models


# Create your models here.
class Jd1(models.Model):
    job_id = models.CharField(primary_key=True, max_length=200)
    description = models.FileField()
    job_title = models.CharField(max_length=200)



    def __str__(self):
        return self.job_id

class Resume(models.Model):
    content=models.FileField()

class Match_per(models.Model):
    job_id=models.ForeignKey(Jd1,on_delete=models.CASCADE)
    resume=models.FileField()
    mp=models.IntegerField(default=0)

    def __str__(self):
        return self.job_id
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models

from django.db import models


class Employee(models.Model):
    username = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    is_admin = models.BooleanField(default=False)

    class Meta:
        db_table = 'employee'

