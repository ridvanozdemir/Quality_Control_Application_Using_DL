%This program is for quality control with webcam and communicating with PLC
%Ridvan Ozdemir

%start webcam
cam = webcam(1);
global capture_image
global image_captured
capture_image=0;
image_captured=0;
close all;
pause(1);

%product quality control

while 1
    % read capture image command from PLC
    capture_image = fread(t,1);
    pause(1);
    if capture_image == 1  
        if image_captured == 0
            pause(1);
            im = snapshot(cam);
            I = im;

            % product detection
            [bboxes, scores] = detect(acfDetector,I,'Threshold',1);
            %Select strongest detection
            [~,idx] = max(scores);
            %Display the detected product
            annotation = acfDetector.ModelName;
            I = insertObjectAnnotation(I,'rectangle',bboxes(idx,:),annotation);
            
            % quality control of detected product
            picture = imresize(I,[227,227]);
            [label,scr] = classify( QNet, picture);
            
            
            figure
            imshow(I)
            title([ num2str(max(scr)), '  ', char(label)]);
            image_captured = 1;
                % send control results to PLC
                if label == 'ok'
                    fwrite(t,4);
                elseif label == 'nok'
                    fwrite(t,6);
                else
                    fwrite(t,0);
                end
            clear cam;
            break
        end
    else
       % image_captured = 0;
    end
end