from .models   import Article 
from rest_framework import serializers


class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model=Article
        fields=['id','title','author']
    # title=serializers.CharField( max_length=50)
    # author=serializers.CharField( max_length=50)      
    # date=serializers.DateField( auto_now=False, auto_now_add=True)
    # email= serializers.EmailField( max_length=254)



    # def create(self,validated_data):
    #     return Article.objects.create(validate_data)

    # def update(self,instance,validate_data):
    #     instance.title = validate_data.get('title',instance.title)
    #     instance.author = validate_data.get('author',instance.author)
    #     instance.date= validate_data.get('date',instance.date)
    #     instance.email = validate_data.get('email',instance.email)
    #     instance.save()
    #     return instance