% Copyright 2018 Suguru KANOGA <s.kanouga@aist.go.jp>
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function feat = getzcfeat(x,deadzone,winsize,wininc,datawin,dispstatus)

if nargin < 6
    if nargin < 5
        if nargin < 4
            if nargin < 3
                winsize = size(x,1);
            end
            wininc = winsize;
        end
        datawin = ones(winsize,1);
    end
    dispstatus = 0;
end

datasize = size(x,1);
Nsignals = size(x,2);
numwin = floor((datasize - winsize)/wininc)+1;

% allocate memory
feat = zeros(numwin,Nsignals);

if dispstatus
    h = waitbar(0,'Computing ZC features...');
end

st = 1;
en = winsize;

for i = 1:numwin
   if dispstatus
       waitbar(i/numwin);
   end
   y = x(st:en,:).*repmat(datawin,1,Nsignals);
   
   y = (y > deadzone) - (y < -deadzone);

   % forces the zeros towards either the positive or negative
   % the filter is chosen so that the most recent +1 or -1 has
   % the most influence on the state of the zero.
   a=1;
   b=exp(-(1:winsize/2))';
   z = filter(b,a,y);
   
   
   z = (z > 0) - (z < -0);
   dz = diff(z);
   
   feat(i,:) = (sum(abs(dz)==2));
   
   st = st + wininc;
   en = en + wininc;
end

if dispstatus
    close(h)
end
