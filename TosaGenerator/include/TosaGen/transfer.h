

#ifndef LLVM_TRANSFER_H
#define LLVM_TRANSFER_H

#endif // LLVM_TRANSFER_H
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/IR/Attributes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "json/json.h"


#include <ctime>
#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>
using namespace mlir;
using namespace std;

class Transfer {
public:
  pair<string, SmallVector<int64_t, 8>>  transferResult(ImplicitLocOpBuilder b);
  pair<string, SmallVector<SmallVector<int64_t, 8>, 8>> transferResult(ImplicitLocOpBuilder b,string opName);
};