
imagefiles = dir('C:\edmem\deep l\tok2018\croplu_dataset\test\*.jpg');     
nfiles = length(imagefiles);    % Number of files found


for ii=1:nfiles
   currentfilename =fullfile('C:\edmem\deep l\tok2018\croplu_dataset\test\', imagefiles(ii).name);
   currentimage = imread(currentfilename);
   images{ii} = currentimage;
   
   picture = images{ii};
   bboxes = detect(acfDetector,picture);
   % bboxes = detect(acfDetector,picture,'Threshold',1);
   % bboxes = step(faceDetector, picture);
   [m,n] = size(bboxes);
   for i=1:1:m
      
        I2 = imcrop(picture,bboxes(i,:));
        I2 = imresize(I2,[227,227]);
        folder = 'C:\edmem\deep l\tok2018\croplu_dataset\sonuc\';
        newimagename = [folder imagefiles(ii).name '_v3' '.jpg'];
        imwrite(I2,newimagename);
   end 
   
end