clear all;close all;clc

acorde = zeros(1,13);
chords_test = ["c","d","dm","e","g"];
chords_main = ["c-major-","d-major-","d-minor-","e-major","g-major"];

for k = 1:5
    
    for n = 1:200
        
        acorde = zeros(1,13);
        filename = sprintf('%s%d.wav',chords_test(k),n);
        
        [y,Fs] = audioread(filename);
        tempofim = length(y);
        [pks,locs] = getFreqs(filename,1,tempofim);
        
        pks = pks/norm(pks);
        %pksDecibels = mag2db(pks);
        
        [notasDoAudio,locs] = getNota(locs);
        
        for i = 1: length(locs)
            
            index = locs(i);
            
            indexNota = find(notasDoAudio == index,1);
            
            %magnitudeDB = pksDecibels(indexNota);
            magnitude = pks(indexNota);
            
            
            acorde(index)= magnitude;
            
            
            
            
        end
        
        %MUDAR ROTULO DO ACORDE
        acorde(13) =k-1;
        dlmwrite('../dataset2_test.csv',acorde,'-append','precision','%0.6f');
        %dlmwrite('../dataset.csv',acorde,'-append')
        %acorde
        
    end
    
    
end

