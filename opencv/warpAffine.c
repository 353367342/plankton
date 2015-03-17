#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "warpAffine.c"
#else

#include <luaT.h>
#include <TH.h>
#include <opencv/cv.h>

static IplImage * libopencv_(torchimage2opencv)(THTensor *source) {
  uchar *destination_data;
  int channels = source->size[0];
  int destination_step;
  CvSize destination_size = cvSize(source->size[2], source->size[1]);
  IplImage * destination = cvCreateImage(destination_size,
                                         IPL_DEPTH_8U,
                                         channels);
  cvGetRawData(destination,
               (uchar**)&destination_data,
               &destination_step,
               &destination_size);
  THTensor *tensor = THTensor_(newContiguous)(source);
  int i = 0;
  int j = 0;
  int k = channels-1;
  uchar *destination_p = destination_data;
  for (j=0;j<destination->nChannels;j++) {
    THTensor *tensor_slice = THTensor_(newSelect)(tensor,0,j);
    for (i=0; i<destination->height; i++) {
      destination_p = destination_data + i*destination_step/sizeof(uchar) + k-j;
      THTensor *row_slice = THTensor_(newSelect)(tensor_slice, 0, i);
      TH_TENSOR_APPLY(real,
                      row_slice,
                      *destination_p = (uchar)(*row_slice_data * 255.0);
                      destination_p = destination_p + destination->nChannels;);
      THTensor_(free)(row_slice);
    }
    THTensor_(free)(tensor_slice);
  }
  THTensor_(free)(tensor);
  return destination;
}

static void libopencv_(opencv2torchimage)(IplImage *source,
                                          THTensor *destination) {
  uchar * source_data;
  int source_step;
  CvSize source_size;
  cvGetRawData(source, (uchar**)&source_data, &source_step, &source_size);
  THTensor_(resize3d)(destination,
                      source->nChannels,
                      source->height,
                      source->width);
  THTensor *tensor = THTensor_(newContiguous)(destination);
  int i = 0; 
  int j = 0;
  int k = source->nChannels-1;
  uchar * source_p = source_data;
  for (j=0;j<source->nChannels;j++){
    THTensor *tensor_slice = THTensor_(newSelect)(tensor,0,j);
    for (i=0; i<source->height; i++) {
      source_p = source_data + i*source_step/sizeof(uchar) + k-j;
      THTensor *row_slice = THTensor_(newSelect)(tensor_slice,0,i);
      TH_TENSOR_APPLY(real,
                      row_slice,
                      *row_slice_data = ((real)(*source_p)) / 255.0;
                      source_p = source_p + source->nChannels;
                      );
      THTensor_(free)(row_slice);
    }
    THTensor_(free)(tensor_slice);
  }
  THTensor_(free)(tensor);
}

static int libopencv_(Main_cvWarpAffine) (lua_State *L) {
  THTensor *source = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *destination = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *warp = luaT_checkudata(L, 3, torch_Tensor);
  THArgCheck(warp->size[0] == 2, 1, "2x3 Tensor expected for transformation");
  THArgCheck(warp->size[1] == 3, 1, "2x3 Tensor expected for transformation");
  IplImage *source_ipl = libopencv_(torchimage2opencv)(source);
  IplImage *destination_ipl = cvCreateImage(cvGetSize(source_ipl),
                                            IPL_DEPTH_8U,
                                            source_ipl->nChannels);
  CvMat* warp_matrix = cvCreateMat(2,3,CV_32FC1);

  THTensor *tensor = THTensor_(newContiguous)(warp);
  float* ptr = warp_matrix->data.fl;
  TH_TENSOR_APPLY(real, tensor, *ptr = (float)*tensor_data; ptr++;);
  THTensor_(free)(tensor);
  cvWarpAffine(source_ipl,
               destination_ipl,
               warp_matrix,
               CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,
               cvScalarAll(0));
  libopencv_(opencv2torchimage)(destination_ipl, destination);
  cvReleaseImage(&source_ipl);
  cvReleaseImage(&destination_ipl);
  cvReleaseMat(&warp_matrix);
  return 0;
}

static const luaL_reg libopencv_(Main__) [] =
{
  {"WarpAffine", libopencv_(Main_cvWarpAffine)}
};

DLL_EXPORT int libopencv_(Main_init) (lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, libopencv_(Main__), "libopencv");
  lua_pop(L,1);
  return 1;
}

#endif