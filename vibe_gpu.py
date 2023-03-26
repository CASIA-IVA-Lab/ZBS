import torch

class ViBeGPU:
    def __init__(self, num_sam=20, min_match=2, radiu=20, rand_sam=16, fake_thres=300):
        self.defaultNbSamples = num_sam 
        self.defaultReqMatches = min_match
        self.defaultRadius = radiu
        self.defaultSubsamplingFactor = rand_sam
        self.background = 0
        self.foreground = 255
        self.fake_thres = fake_thres

    def __buildNeighborArray(self, img):
        height, width = img.shape
        ramoff_xy = torch.randint(-1, 2, size=(2, self.defaultNbSamples, height, width))
        
        xr_ = torch.tile(torch.arange(width), (height, 1))
        yr_ = torch.tile(torch.arange(height), (width, 1)).T

        xyr_ = torch.zeros((2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_

        xyr_ = xyr_ + ramoff_xy

        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_

        xyr = xyr_.numpy().astype(int)
        self.samples = torch.from_numpy(img[xyr[0, :, :, :], xyr[1, :, :, :]]).cuda()

    def ProcessFirstFrame(self, img):
        self.__buildNeighborArray(img)
        self.fgCount = torch.zeros(img.shape).cuda()  
        self.fgMask = torch.zeros(img.shape).cuda()

    def Update(self, img):
        height, width = img.shape
        img = torch.from_numpy(img).cuda()
        dist = torch.abs((self.samples.float() - img.float()).int())
        dist[dist < self.defaultRadius] = 1
        dist[dist >= self.defaultRadius] = 0
        matches = torch.sum(dist, axis=0)
        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background
        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0
        fakeFG = self.fgCount > self.fake_thres
        matches[fakeFG] = False
        
        upfactor = torch.randint(self.defaultSubsamplingFactor, size=img.shape, device='cuda')
        upfactor[matches] = 100  
        upSelfSamplesInd = torch.where(upfactor == 0)  
        upSelfSamplesPosition = torch.randint(
            self.defaultNbSamples,
            size=upSelfSamplesInd[0].shape,
            device='cuda') 
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0],
                  upSelfSamplesInd[1])
        self.samples[samInd] = img[upSelfSamplesInd] 
        
        upfactor = torch.randint(self.defaultSubsamplingFactor,
                                     size=img.shape, device='cuda')
        upfactor[matches] = 100 
        upNbSamplesInd = torch.where(upfactor == 0)
        nbnums = upNbSamplesInd[0].shape[0]
        ramNbOffset = torch.randint(-1, 2, size=(2, nbnums), device='cuda')
        nbXY = torch.stack(upNbSamplesInd)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = torch.randint(self.defaultNbSamples, size=(nbnums,), device='cuda')
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        self.samples[nbSamInd] = img[upNbSamplesInd]

    def getFGMask(self):
        return self.fgMask