
#include "TosaGen/utils.h"
#include "TosaGen/create.h"
#include "TosaGen/opinfo.h"
#include "TosaGen/transfer.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include <map>
#include <random>

using namespace mlir;
using namespace std;

extern Utils genUtils;
extern InfoGen infogen;
extern Create create;
extern opInfo info;
extern Transfer transfer;

DenseMap<int, SmallVector<string,8>>  opsDim;
DenseMap<int, SmallVector<string,8>>  opsType;

std::random_device rd3;
std::mt19937 gen3(rd3());
#define DEBUG_TYPE "user-define-pass"

namespace {
struct MIXPass : public PassWrapper<MIXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MIXPass)
  SmallVector<Value> insertTosaOp(ImplicitLocOpBuilder b,Location loc,Value conOp,func::FuncOp func, SmallVector<Value> valuePool);
  Value Conversion(ImplicitLocOpBuilder b,Location loc,Value insOp,Value conOp,func::FuncOp f);
  Value shapeConversion(ImplicitLocOpBuilder b,Location loc,Value insOp,Value conOp,func::FuncOp f);
  Value createReshapeOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                        llvm::SmallVector<int64_t , 8> newShape);
  Value createConstOp(ImplicitLocOpBuilder b, Location loc,  string et, llvm::SmallVector<int64_t , 8> concatShape,func::FuncOp f);
  Value createConcatOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                       Value constOp);
  Value createConstOpForMixConvert(ImplicitLocOpBuilder b, Location loc,string type, llvm::SmallVector<int64_t , 8> newShape,func::FuncOp f);
  Value createCastOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                     Value constOp);
  Value createSliceOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                      llvm::SmallVector<int64_t , 8> con_Shape);
  Value createReduceOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                       int axis);
  void runOnOperation() override {



    cout << "Mix"<< endl;

    auto f = getOperation();

    func::FuncOp::iterator fi = f.getBlocks().begin();
    func::FuncOp::iterator fe = f.getBlocks().end();

    SmallVector<Block *,16> blockStack;

    while(fi != fe) {
      blockStack.push_back(&(*fi));
      fi++;
    }

    std::uniform_int_distribution<unsigned> randomNum(0, blockStack.size()-1);

    Block * bb;
    do{
      bb = blockStack[randomNum(gen3)];
    }while(&(*bb->begin())==bb->getTerminator());

    int times = 1;
    while (times--) {

      Location loc = bb->begin()->getLoc();
      ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(loc, bb);
      b.setInsertionPointToStart(bb);

      Operation * returnOp;

      SmallVector<Operation *> rawOps;
      llvm::SmallVector<string> specialOp = {"memref.load","tensor.extract","memref.store"};
      getOperation().walk([&](Operation *op) {
        if (!mlir::isa<func::ReturnOp>(op)){
          if (op->getNumResults()==1){
            if(op->getResult(0).getType().isa<TensorType>()){
              rawOps.push_back(op);
            }
          }
        } else{
          returnOp = op;
        }

      });


      unsigned int r ;
      unsigned int n ;
      SmallVector<Value> valuePool;
      if(rawOps.empty())
        break;
      else{
        if (rawOps.size() == 1){
          r = 0;
          valuePool.push_back(rawOps[r]->getResult(0));
        }
        else{
          r = genUtils.genRandomN(1,rawOps.size()-1);
          for(auto op : rawOps){
            valuePool.push_back(rawOps[r]->getResult(0));
            if(op==rawOps[r])
              break;
          }
        }
      }

      auto curOp = rawOps[r];
      cout<<"当前插入点："<<endl;
      curOp->dump();
      b.setInsertionPointAfter(curOp);
      Value conOp = curOp->getResult(0);

      if (conOp.getType().isa<MemRefType>()){
        cout<<"============ memref to tensor"<<endl;
        conOp = b.create<bufferization::ToTensorOp>(loc, conOp);

      }

      cout<<"=============new tosa insOp: "<<endl;
      SmallVector<Value> allInsOps = insertTosaOp(b,loc,conOp,f, valuePool);
      auto new_tosa_insOp = allInsOps.back();
      new_tosa_insOp.dump();
      if(allInsOps.empty())
        break;

      Value newInsOp = Conversion(b,loc,new_tosa_insOp,conOp,f);
      cout<<"=============converted op"<<endl;
      newInsOp.dump();


      SmallPtrSet<Operation *,4> exceptUser;
      for(auto op : allInsOps){
        exceptUser.insert(op.getDefiningOp());
      }


      conOp.replaceAllUsesExcept(newInsOp,exceptUser);


      f.dump();

      do{
        bb = blockStack[randomNum(gen3)];
      }while(&(*bb->begin())==bb->getTerminator());
    }

  };



  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect,scf::SCFDialect,arith::ArithDialect,mlir::bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const final { return "Mix"; }
  StringRef getDescription() const final { return "Mix pass"; }
};

}
Value MIXPass::shapeConversion(ImplicitLocOpBuilder b,Location loc,Value insOp,Value conOp,func::FuncOp func){
  pair<string, SmallVector<int64_t, 8>> con_Info = infogen.getResultInfo(conOp);
  pair<string, SmallVector<int64_t, 8>> ins_Info = infogen.getResultInfo(insOp);
  string con_type = con_Info.first;
  string ins_type = ins_Info.first;
  llvm::SmallVector<int64_t , 8> con_shape = con_Info.second;
  llvm::SmallVector<int64_t , 8> ins_shape = ins_Info.second;

  Value newInsOp;
  pair<string, SmallVector<int64_t,8>> result_Info;
  result_Info = con_Info;

  int e1 = insOp.getType().cast<ShapedType>().getNumElements();
  int e2 = conOp.getType().cast<ShapedType>().getNumElements();
  float i = (float)e1 / e2;
  if (i==1) {
    newInsOp = createReshapeOp(b,loc,insOp,con_shape);
  }
  else if (i<1){
    SmallVector<int> index;
    int pro = 1;
    llvm::SmallVector<int64_t , 8> concatShape;
    if(ins_shape.size()==con_shape.size()){
      for(int i = 0;i<ins_shape.size();i++){
        if(ins_shape[i]==con_shape[i]){
          index.push_back(i);
          concatShape.push_back(ins_shape[i]);
        }
        else{
          pro = (con_shape[i]/ins_shape[i])*pro;
          concatShape.push_back(con_shape[i]-ins_shape[i]);
        }
      }
    }

    if(!index.empty() && con_shape.size()-index.size()==1){
      float en = e2/e1;
      if(e2/e1 - (int)e2/e1==0){
        if ((int)en==pro){
          cout<<"expand shape"<<endl;
        }

        Value constOp = createConstOp( b,  loc,  ins_type,  concatShape,func);
        newInsOp = createConcatOp(b,loc,insOp,constOp);
      }
    }else{
      llvm::SmallVector<int64_t , 8> newShape;
      for(auto x :con_shape){
        if(*con_shape.begin()==x)
          newShape.push_back(e2-e1);
        else
          newShape.push_back(1);
      }

      Value padOp = createConstOpForMixConvert(b,loc,ins_type,newShape,func);


      newShape[0]=e1;
      Value flattenOp = createReshapeOp(b,loc,insOp,newShape);
      flattenOp.dump();

      Value concatOp = createConcatOp(b,loc,padOp,flattenOp);

      newInsOp = createReshapeOp(b,loc,concatOp,con_shape);
    }
  }

  return newInsOp;
}


Value MIXPass::Conversion(ImplicitLocOpBuilder b,Location loc,Value insOp,Value conOp,func::FuncOp f){
  Value SCOp;
  Value TCOp;
  if (insOp.getType().cast<ShapedType>().getShape()!=conOp.getType().cast<ShapedType>().getShape()){
    Value SCOp = shapeConversion(b,loc,insOp,conOp,f);
    insOp = SCOp;
  }

  if (insOp.getType().cast<ShapedType>().getElementType()!=conOp.getType().cast<ShapedType>().getElementType()){
    Value TCOp = createCastOp(b,loc,insOp,conOp);
    insOp = TCOp;
  }
  return insOp;
}


SmallVector<Value> MIXPass::insertTosaOp(ImplicitLocOpBuilder b,Location loc,Value conOp,func::FuncOp func, SmallVector<Value> valuePool){



  llvm::ArrayRef<StringRef> opPool = {"pad","matmul","transpose_conv2d","identity","arithmetic_right_shift", "negate", "transpose", "concat","abs","equal","bitwise_not", "ceil","clz","clamp","exp","floor","log","logical_not",
                                          "reciprocal","rsqrt","sigmoid","tanh","argmax","reduce_all","reduce_any","reduce_max","reduce_min","reduce_prod",
                                          "reduce_sum","reverse","add","bitwise_and","bitwise_or","bitwise_xor","div","greater","greater_equal",
                                          "logical_and","logical_left_shift","logical_or","logical_right_shift","logical_xor","mul","maximum","minimum","pow","sub",
                                          "slice","tile","conv2d","conv3d","depthwise_conv2d","avg_pool2d","max_pool2d","reshape","cast"};

  genUtils.analyseAllOp(opPool);

  SmallVector<Value> allInsOps = {};

  int opNum = genUtils.genRandomN(3, 10);;
  Value newOp;
  for(int i=0 ; i < opNum ; i++) {
    string selectedOp = genUtils.getNextOp(conOp);

    cout<<"selectedOp  "<<selectedOp<<endl;
    if(selectedOp==" ")
      return allInsOps;

    infogen.initInfo(selectedOp);
    infogen.addInputType(b, conOp);

    Value v = genUtils.skipMatch(valuePool);
    if(v==nullptr){
      infogen.addInputs(b, loc, func, conOp);
    }else{
      info.inputs.push_back(conOp);
      info.inputs.push_back(v);
      info.inputType={};
      info.inputType.push_back(conOp.getType());
      info.inputType.push_back(v.getType());
    }

    infogen.addAttrs(b, loc);
    infogen.addResult(b);
    newOp =  create.createOp(b,loc);
    allInsOps.push_back(newOp);
    conOp=newOp;
  }

  return allInsOps;
}

Value MIXPass::createReshapeOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                               llvm::SmallVector<int64_t , 8> newShape) {

  string opname = "reshape";
  infogen.initInfo(opname);

  info.inputType.push_back(insOp.getType());
  info.inputs.push_back(insOp);


  SmallVector<NamedAttribute> namedAttrs;
  namedAttrs.push_back(b.getNamedAttr("new_shape",b.getDenseI64ArrayAttr(newShape)));
  info.attrs = namedAttrs;
  infogen.addResult(b);


  Value reshapeOp = create.createOpWithOneAttr(b, loc);
  return reshapeOp;
}

Value MIXPass::createConstOp(ImplicitLocOpBuilder b, Location loc, string et, llvm::SmallVector<int64_t , 8> concatShape,func::FuncOp funcOp){
  Value constOp;
  if (genUtils.getElementNum(concatShape) > 10000){
    string opname = "log";
    infogen.initInfo(opname);
    llvm::SmallVector<Type , 8> args;
    args.push_back(genUtils.genTensorType(b,concatShape,et));
    create.insertFuncArg(b,funcOp,args);
    infogen.addInputs(b,loc,funcOp);
    constOp = info.inputs[0];
  }
  else{
    DenseIntOrFPElementsAttr inputAttr = genUtils.getDenseAttr(b,et,concatShape);
    constOp = b.create<tosa::ConstOp>(loc, inputAttr.getType(), inputAttr);
  }
  return constOp;
}

Value MIXPass::createConcatOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                              Value constOp){

  string opname = "concat";
  infogen.initInfo(opname);

  info.inputType.push_back(insOp.getType());
  info.inputType.push_back(constOp.getType());
  info.inputs.push_back(insOp);
  info.inputs.push_back(constOp);

  infogen.addAttrs(b, loc);

  infogen.addResult(b);

  Value concatOp = create.createOpWithOneAttr(b, loc);
  return concatOp;
}

Value MIXPass::createConstOpForMixConvert(ImplicitLocOpBuilder b, Location loc,string et,  llvm::SmallVector<int64_t , 8> inputShape,func::FuncOp funcOp){

  Value constOp;
  if (genUtils.getElementNum(inputShape) > 10000){
    string opname = "log";
    infogen.initInfo(opname);
    llvm::SmallVector<Type , 8> args;
    args.push_back(genUtils.genTensorType(b,inputShape,et));
    create.insertFuncArg(b,funcOp,args);
    infogen.addInputs(b,loc,funcOp);
    constOp = info.inputs[0];
  }
  else {

    DenseIntOrFPElementsAttr inputAttr =
        genUtils.getDenseAttr(b, et, inputShape);
    constOp =
        b.create<tosa::ConstOp>(loc, inputAttr.getType(), inputAttr);
  }
  return constOp;
}

Value MIXPass::createCastOp(ImplicitLocOpBuilder b,Location loc,Value insOp,Value conOp){

  string opname = "cast";
  infogen.initInfo(opname);

  info.inputType.push_back(insOp.getType());
  info.inputs.push_back(insOp);
  info.resultType = conOp.getType();

  Value castOp = create.createOpWithNoAttr(b, loc);
  return castOp;
}

Value MIXPass::createReduceOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                              int axis){

  string opname = "reduce_sum";
  infogen.initInfo(opname);

  info.inputType.push_back(insOp.getType());
  info.inputs.push_back(insOp);

  SmallVector<NamedAttribute> namedAttrs;
  namedAttrs.push_back(b.getNamedAttr("axis", b.getI64IntegerAttr(axis)));
  info.attrs = namedAttrs;
  infogen.addResult(b);

  Value concatOp = create.createOpWithOneAttr(b, loc);
  return concatOp;
}


Value MIXPass::createSliceOp(mlir::ImplicitLocOpBuilder b, mlir::Location loc, mlir::Value insOp,
                             llvm::SmallVector<int64_t , 8> con_shape){

  string opname = "slice";
  infogen.initInfo(opname);

  info.inputType.push_back(insOp.getType());
  info.inputs.push_back(insOp);
  llvm::SmallVector<int64_t , 8> startVec;
  for(auto x:con_shape){
    startVec.push_back(0);
  }

  SmallVector<NamedAttribute> namedAttrs;
  namedAttrs.push_back(b.getNamedAttr("start",b.getDenseI64ArrayAttr(startVec)));
  namedAttrs.push_back(b.getNamedAttr("size",b.getDenseI64ArrayAttr(con_shape)));
  info.attrs = namedAttrs;

  infogen.addResult(b);


  Value sliceOp = create.createOpWithMulAttrs(b, loc);
  return sliceOp;
}

namespace mlir {
void registerMixMutatePass() { PassRegistration<MIXPass>(); }
}
