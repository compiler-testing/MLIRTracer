
#include "TosaGen/transfer.h"

#include "TosaGen/opinfo.h"
#include "TosaGen/utils.h"
using namespace std;
extern Utils genUtils;
extern InfoGen infogen;
extern opInfo info;
extern Transfer transfer;
extern SmallVector<int, 8> perms_const_int;

pair<string, SmallVector<int64_t, 8>> Transfer::transferResult(ImplicitLocOpBuilder b) {

  auto opConstraint = info.Constraint;
  Json::Value result = opConstraint["RESULT"];

  string elementType;
  string typeStr = result[0]["TYPE"].asString();
  if(typeStr=="any" || typeStr=="float" || typeStr=="int" || typeStr=="floatOruint")
      elementType = genUtils.getTensorType(info.inputType[0]);
  else
      elementType = genUtils.getTypestr(typeStr);

  SmallVector<int64_t, 8> resultShape ;
  if(info.needBroadcast==1) {
    resultShape = genUtils.genBroadcastShape();
  }else{
    resultShape = genUtils.getTensorShape(info.inputType[0]);
  }

  if(info.opName=="cast") {
    llvm::ArrayRef<string> types= {"i1", "i8", "i16", "i32", "i64", "f32"};
    SmallVector<string,8> candidates;
    for(auto type:types) {
      if(type!=elementType)
        candidates.push_back(type);
    }
    elementType = types[genUtils.genRandomN(0, candidates.size() - 1)];
  }


  if (info.opName == "reshape") {
    SmallVector<int64_t,8> newShape;
    auto array = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    for(auto a:array){
      newShape.push_back(a);
    }

    resultShape = newShape;
  }


  if (info.opName == "reduce_all") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    resultShape = genUtils.getTensorShape(info.inputType[0]);
    resultShape[axis] = 1;
  }

  if (info.opName == "reduce_any") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    resultShape = genUtils.getTensorShape(info.inputType[0]);
    resultShape[axis] = 1;
  }
  if (info.opName == "reduce_max") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    resultShape = genUtils.getTensorShape(info.inputType[0]);
    resultShape[axis] = 1;
  }
  if (info.opName == "reduce_min") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    resultShape = genUtils.getTensorShape(info.inputType[0]);
    resultShape[axis] = 1;
  }
  if (info.opName == "reduce_prod") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    resultShape = genUtils.getTensorShape(info.inputType[0]);
    resultShape[axis] = 1;
  }

  if (info.opName == "reduce_sum") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    resultShape = genUtils.getTensorShape(info.inputType[0]);
    resultShape[axis] = 1;
  }

  if (info.opName == "concat") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    int Dim = info.inputType[0].cast<ShapedType>().getRank();
    if (axis >= Dim) {
      axis = genUtils.genRandomN(0, Dim - 1);
      info.attrs[0].setValue(b.getI64IntegerAttr(axis));
    }
    SmallVector<int64_t, 8> shape1 =
        genUtils.getTensorShape(info.inputType[0]);
    SmallVector<int64_t, 8> shape2 =
        genUtils.getTensorShape(info.inputType[1]);
    resultShape[axis] = shape1[axis] + shape2[axis];
  }

  if (info.opName == "argmax") {
    auto axis =
        info.attrs[0].getValue().cast<IntegerAttr>().getValue().getSExtValue();
    SmallVector<int64_t, 8> shape =
        genUtils.getTensorShape(info.inputType[0]);
    auto SI = shape.begin();
    for (int j = 0; j < shape.size(); j++) {
      if (j == axis)
        break;
      else
        SI++;
    }
    shape.erase(SI);
    resultShape = shape;
  }

  if(info.opName == "arithmetic_right_shift"){
    auto round = info.attrs[0].getValue().cast<BoolAttr>().getValue();
      SmallVector<int64_t, 8> shape1 =
              genUtils.getTensorShape(info.inputType[0]);
      SmallVector<int64_t, 8> shape2 =
              genUtils.getTensorShape(info.inputType[1]);
      if(round == 1)
          resultShape = shape1;
      else
          resultShape = shape1;
  }

  if(info.opName == "tile"){
      auto multiple = info.attrs[0].getValue().cast<DenseI64ArrayAttr >().asArrayRef();
      SmallVector<int64_t, 8> shape1 =
              genUtils.getTensorShape(info.inputType[0]);
      SmallVector<int64_t ,8> shape;
      for(int i=0;i < info.inputType[0].cast<ShapedType>().getRank();i++){
          int m = multiple[i];
          shape.push_back(shape1[i]*m);
      }
      resultShape = shape;
  }

  if(info.opName == "transpose"){
      SmallVector<int64_t, 8> input_shape =
              genUtils.getTensorShape(info.inputType[0]);
      SmallVector<int64_t ,8> result_shape;
      for(auto i : perms_const_int){
          result_shape.push_back(input_shape[i]);
      }
      resultShape =  result_shape ;
  }

  if (info.opName == "conv2d"){

    SmallVector<int64_t,8> input = genUtils.getTensorShape(info.inputType[0]) ;
    SmallVector<int64_t,8> weight = genUtils.getTensorShape(info.inputType[1]) ;
    SmallVector<int64_t,8> bias = genUtils.getTensorShape(info.inputType[2]) ;

    auto padElem = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto strideElem = info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto dilationElem = info.attrs[2].getValue().cast<DenseI64ArrayAttr>().asArrayRef();

    SmallVector<int64_t, 8> shape;
    shape.push_back(input[0]);
    shape.push_back(div(input[1]-1+padElem[0]+padElem[1]-(weight[1]-1)*dilationElem[0],strideElem[0]).quot+1);
    shape.push_back(div(input[2]-1+padElem[2]+padElem[3]-(weight[2]-1)*dilationElem[1],strideElem[1]).quot+1);
    shape.push_back(bias[0]);
    resultShape = shape;
  }

  if (info.opName=="conv3d"){

    SmallVector<int64_t,8> input = genUtils.getTensorShape(info.inputType[0]) ;
    SmallVector<int64_t,8> weight = genUtils.getTensorShape(info.inputType[1]) ;
    SmallVector<int64_t,8> bias = genUtils.getTensorShape(info.inputType[2]) ;

    auto padElem = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto strideElem = info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto dilationElem = info.attrs[2].getValue().cast<DenseI64ArrayAttr>().asArrayRef();

    SmallVector<int64_t, 8> shape;
    shape.push_back(input[0]);
    shape.push_back(div(input[1]-1+padElem[0]+padElem[1]-(weight[1]-1)*dilationElem[0],strideElem[0]).quot+1);
    shape.push_back(div(input[2]-1+padElem[2]+padElem[3]-(weight[2]-1)*dilationElem[1],strideElem[1]).quot+1);
    shape.push_back(div(input[3]-1+padElem[4]+padElem[5]-(weight[3]-1)*dilationElem[2],strideElem[2]).quot+1);
    shape.push_back(bias[0]);
    resultShape = shape;
  }

  else if (info.opName=="transpose_conv2d"){
    SmallVector<int64_t,8> input = genUtils.getTensorShape(info.inputType[0]) ;
    SmallVector<int64_t,8> weight = genUtils.getTensorShape(info.inputType[1]) ;
    SmallVector<int64_t,8> bias = genUtils.getTensorShape(info.inputType[2]) ;


    auto out_pad = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto strideElem = info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto dilationElem = info.attrs[2].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto out_shape = info.attrs[3].getValue().cast<DenseI64ArrayAttr>().asArrayRef();

    SmallVector<int64_t, 8> shape;
    shape.push_back(input[0]);
    shape.push_back((input[1]-1)*strideElem[0]+out_pad[0]+out_pad[1]+weight[1]);
    shape.push_back((input[2]-1)*strideElem[1]+out_pad[2]+out_pad[3]+weight[2]);
    shape.push_back(bias[0]);
    resultShape = shape;

  }

  if (info.opName=="depthwise_conv2d"){
    SmallVector<int64_t,8> input = genUtils.getTensorShape(info.inputType[0]) ;
    SmallVector<int64_t,8> weight = genUtils.getTensorShape(info.inputType[1]) ;
    SmallVector<int64_t,8> bias = genUtils.getTensorShape(info.inputType[2]) ;
    auto padElem = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto strideElem = info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto dilationElem = info.attrs[2].getValue().cast<DenseI64ArrayAttr>().asArrayRef();

    SmallVector<int64_t, 8> shape;
    shape.push_back(input[0]);
    shape.push_back(div(input[1]-1+padElem[0]+padElem[1]-(weight[0]-1)*dilationElem[0],strideElem[0]).quot+1);
    shape.push_back(div(input[2]-1+padElem[2]+padElem[3]-(weight[1]-1)*dilationElem[1],strideElem[1]).quot+1);
    shape.push_back(bias[0]);
    resultShape = shape;
  }
  if(info.opName == "scatter"){
      SmallVector<int64_t, 8> shape1 = genUtils.getTensorShape(info.inputType[0]);
      resultShape = shape1;
  }

  if (info.opName=="max_pool2d" || info.opName=="avg_pool2d"){
    SmallVector<int64_t,8> input = genUtils.getTensorShape(info.inputType[0]) ;

    auto kernelElem = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto strideElem = info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto padElem = info.attrs[2].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    SmallVector<int64_t, 8> shape;
    shape.push_back(input[0]);
    shape.push_back(div(input[1]+padElem[0]+padElem[1]-kernelElem[0],strideElem[0]).quot+1);
    shape.push_back(div(input[2]+padElem[2]+padElem[3]-kernelElem[1],strideElem[1]).quot+1);
    shape.push_back(input[3]);
    resultShape = shape;
  }
  else if (info.opName=="slice"){
      SmallVector<int64_t, 8> newShape;
      auto array =
              info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
      for (auto a : array) {
          newShape.push_back(a);
      }
      resultShape = newShape;
  }
  else if (info.opName=="clamp"){
      SmallVector<int64_t, 8> shape =
              genUtils.getTensorShape(info.inputType[0]);
      resultShape = shape;
  }else if(info.opName == "select"){
      SmallVector<int64_t, 8> shape1 =
                    genUtils.getTensorShape(info.inputType[0]);
      auto rank1 = info.inputType[0].cast<ShapedType>().getRank();
      SmallVector<int64_t, 8> shape2 =
                    genUtils.getTensorShape(info.inputType[1]);
      auto rank2 = info.inputType[1].cast<ShapedType>().getRank();
      SmallVector<int64_t, 8> shape3 =
                    genUtils.getTensorShape(info.inputType[2]);
      auto rank3 = info.inputType[2].cast<ShapedType>().getRank();
      int64_t n = (rank1>=rank2)?rank1:rank2;
      if(n == rank1){
          resultShape = shape1;
      }else if(n == rank2){
          resultShape = shape2;
      }
      if(rank1 == rank2){
          auto type1 = info.inputType[0].cast<ShapedType>().getShape();
          auto type2 = info.inputType[1].cast<ShapedType>().getShape();
          if(type1[0]&&type2[0]&&type1[0]>=type2[0]){
              n = rank1;
              resultShape = shape1;
          }
          else{
              n = rank2;
              resultShape = shape2;
          }
      }
      if(n<rank3){
          resultShape = shape3;
      }
      else if(n == rank3){
          if(n == rank1){
              auto type1 = info.inputType[0].cast<ShapedType>().getShape();
              auto type2 = info.inputType[2].cast<ShapedType>().getShape();
              if(type1[0]&&type2[0]&&type1[0]>type2[0])
                  resultShape = shape1;
              else
                  resultShape = shape2;
          }
          else{
              auto type1 = info.inputType[1].cast<ShapedType>().getShape();
              auto type2 = info.inputType[2].cast<ShapedType>().getShape();
              if(type1[0]&&type2[0]&&type1[0]>=type2[0])
                  resultShape = shape1;
              else
                  resultShape = shape2;
          }
      }
  }
  else if (info.opName=="resize"){
    SmallVector<int64_t, 8> inputShape = genUtils.getTensorShape(info.inputType[0]);
    SmallVector<int64_t, 8> outputShape;
    outputShape.push_back(inputShape[0]) ;
    int64_t inputHeight = inputShape[1];
    int64_t inputWidth = inputShape[2];
    auto scaleInt  = info.attrs[0].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto offsetInt = info.attrs[1].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    auto borderInt = info.attrs[2].getValue().cast<DenseI64ArrayAttr>().asArrayRef();
    outputShape.push_back(inputShape[0]) ;
    outputShape.push_back((((inputHeight - 1) * scaleInt[0] - offsetInt[0] + borderInt[0]) / scaleInt[1]) + 1);
    outputShape.push_back((((inputWidth - 1) * scaleInt[2] - offsetInt[1] + borderInt[1]) / scaleInt[3]) + 1);
    outputShape.push_back(inputShape[3]) ;
    resultShape = outputShape;
  }
  else if (info.opName=="rescale"){
      SmallVector<int64_t, 8> shape =
              genUtils.getTensorShape(info.inputType[0]);
      resultShape = shape;
  }else if(info.opName=="gather"){
      SmallVector<int64_t, 8> shape;
      auto shape1 = info.inputType[0].cast<ShapedType>().getShape();
      auto shape2 = info.inputType[1].cast<ShapedType>().getShape();
      shape.push_back(shape1[0]);
      shape.push_back(shape2[1]);
      shape.push_back(shape1[2]);
      resultShape = shape;
   }else if(info.opName=="matmul"){
          SmallVector<int64_t, 8> shape;
          auto shape1 = info.inputType[0].cast<ShapedType>().getShape();
          auto shape2 = info.inputType[1].cast<ShapedType>().getShape();
          shape.push_back(shape1[0]);
          shape.push_back(shape1[1]);
          shape.push_back(shape2[2]);
          resultShape = shape;
       }
  else if(info.opName=="table"){
      SmallVector<int64_t,8> shape1 = genUtils.getTensorShape(info.inputType[0]) ;
      resultShape = shape1;
  }
   else if(info.opName=="fully_connected"){
        auto shape1 = genUtils.getTensorShape(info.inputType[0]);
        auto shape2 = genUtils.getTensorShape(info.inputType[1]);
         SmallVector<int64_t, 8> shape;
         shape.push_back(shape1[0]);
         shape.push_back(shape2[0]);
         resultShape = shape;
       }


  return {elementType,resultShape};
}


pair<string, SmallVector<SmallVector<int64_t, 8>, 8>> Transfer::transferResult(ImplicitLocOpBuilder b, string opName) {
  SmallVector<SmallVector<int64_t, 8>, 8> resultShape ;
  string elementType = genUtils.getTensorType(info.inputType[0]);
  if (opName == "rfft2d"){
    SmallVector<int64_t, 8> inputShape = genUtils.getTensorShape(info.inputType[0]);
    SmallVector<int64_t, 8> outputShape;
    outputShape[0] = inputShape[0];
    outputShape[1] = inputShape[1];
    int64_t inWidth = inputShape[2];
    outputShape[2] = inWidth / 2 + 1;
    resultShape.push_back(outputShape);
    resultShape.push_back(outputShape);
  }
  return {elementType,resultShape};
}