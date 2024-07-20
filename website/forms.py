from django import forms
class mf(forms.Form):
    job_description=forms.CharField(widget=forms.Textarea(attrs={'class':'form-control','rows':'4'}))
    resume=forms.CharField(widget=forms.Textarea(attrs={'class':'form-control','rows':'3'}))