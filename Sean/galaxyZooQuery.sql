SELECT 
gz.objid, gz.p_el_debiased, gz.p_cs_debiased,
gal.probPSF, 
gal.petroRad_u, gal.petroRad_g, gal.petroRad_r, gal.petroRad_i, gal.petroRad_z,
gal.q_u,gal.q_g,gal.q_r,gal.q_i,gal.q_z,
gal.u_u,gal.u_g,gal.u_r,gal.u_i,gal.u_z,
gal.mE1_u,gal.mE1_g,gal.mE1_r,gal.mE1_i,gal.mE1_z,
gal.mE2_u,gal.mE2_g,gal.mE2_r,gal.mE2_i,gal.mE2_z,
gal.mRrCc_u,gal.mRrCc_g,gal.mRrCc_r,gal.mRrCc_i,gal.mRrCc_z,
gal.mCr4_u,gal.mCr4_g,gal.mCr4_r,gal.mCr4_i,gal.mCr4_z,
gal.deVRad_u,gal.deVRad_g,gal.deVRad_r,gal.deVRad_i,gal.deVRad_z,
gal.deVAB_u,gal.deVAB_g,gal.deVAB_r,gal.deVAB_i,gal.deVAB_z,
gal.expRad_u,gal.expRad_g,gal.expRad_r,gal.expRad_i,gal.expRad_z,
gal.expAB_u,gal.expAB_g,gal.expAB_r,gal.expAB_i,gal.expAB_z,
gal.modelFlux_u,gal.modelFlux_g,gal.modelFlux_r,gal.modelFlux_i,gal.modelFlux_z,
gal.u,gal.g,gal.r,gal.i,gal.z,
gal.extinction_u,gal.extinction_g,gal.extinction_r,gal.extinction_i,gal.extinction_z,
gal.dered_u,gal.dered_g,gal.dered_r,gal.dered_i,gal.dered_z,
gal.psffwhm_u,gal.psffwhm_g,gal.psffwhm_r,gal.psffwhm_i,gal.psffwhm_z,
gal.u-gal.g,gal.g-gal.r, gal.r-gal.i, gal.i-gal.z,
soa.z, soa.velDisp

FROM zooSpec AS gz
INTO GalaxyZooData
JOIN Galaxy AS gal ON gz.objid = gal.obji
JOIN SpecObjAll AS soa ON gz.specObjid = soa.specObjid
