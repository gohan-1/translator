from django.db import models



# Create your models here.

class  Article(models.Model):
    title=models.CharField( max_length=50)
    author=models.CharField( max_length=50)    
    date=models.DateField( auto_now=False, auto_now_add=True)
    email= models.EmailField( max_length=254)
    # verbose_name = _("")
    # verbose_name_plural = _("s")

    def __str__(self):
        return self.title

#     def get_absolute_url(self):
#         return reverse("_detail", kwargs={"pk": self.pk})
# )