# Health-Authority-Classifer

Problem: In order to recognize the Health Authority sending the document my company was using the below mentioned approach:-

 1) OCR document (15-20 pages)
 2) Text extraction
 3) Passing the text to the text classifier
 4) All the above steps took a lot of time and was pretty slow
 
 
Solution:
  1) Just because it is a text document doesnt mean that it can only be solved using NLP
  2) An image is worth a 1000 words, always remember ;)
  3) The person who suggested the NLP route didnt explore the data properly 
  4) Upon exploration i realized that the first page of every document has the logo of the particular health authority and there was no     
  need to actually such a long route in order to classify
  5) I converted the first page of every document into an image and simply trained a CNN using VGG16 (benefits of transfer learning)
  6) Time needed to be processed and classified improved significantly
  
  
  
