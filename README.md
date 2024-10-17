
# Analysis Framework for Identifying Brand Differentiating Messages (BDM) in Commercials
<div style="text-align: center;">
  <img src="./Resources/images/SuperBowl.png" alt="Super Bowl" width="300"/>
</div>

This Project aims to provide a Framework which can be used to analyse a commercial and identify whether or not this commercial contains a Brand Differentiating Message (BDM)

---

# Structure

The [Resources](./Resources) Directory should Contain the Raw Commercial Files from 2013 to 2022. In order to keep the filesize of this repo small, thees have been omitted and must be obtained separately and need to be placed in the folder structure manually like so:

```sh
Resources/
  └── Ads/
    └── ADs_IG_2013/
      ├── AD0252.mp4
      ├── AD0253.mp4
      ├── AD0254.mp4
      ├── ...
    └── ADs_IG_2014/
      ├── AD0301.mp4
      ├── AD0302.mp4
      ├── ...
```


This [Spreadsheet](./Resources/CSV/SB_AD_LIST__2013-2022.xlsx) contains some metadata and indicates whether or not the add contains a BDM based on human feedback.