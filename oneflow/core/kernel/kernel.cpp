#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::Init(const KernelConf& kernel_conf) {
  kernel_conf_ = kernel_conf;
  VirtualKernelInit();
}

void Kernel::InitModelBlobs(
    const KernelCtx& ctx, const ParallelContext& parallel_ctx,
    const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t part_id = -1;
  int32_t part_num = -1;
  if (parallel_ctx.policy() == kDataParallel) {
    part_id = 0;
    part_num = 1;
  } else if (parallel_ctx.policy() == kModelParallel) {
    part_id = parallel_ctx.parallel_id();
    part_num = parallel_ctx.parallel_num();
  } else {
    UNEXPECTED_RUN();
  }
  std::string model_load_dir = kernel_conf().op_conf().model_load_dir();
  if (model_load_dir == "" && snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op_conf().name());
  }
  if (model_load_dir == "") {
    uint32_t random_seed = reinterpret_cast<uint64_t>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    InitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
  } else {
    InitModelBlobsWithDir(ctx, part_id, part_num, model_load_dir, BnInOp2Blob);
  }
}

void Kernel::InitModelTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

const std::string& Kernel::Lbn4BnInOp(const std::string& bn_in_op) const {
  return kernel_conf_.bn_in_op2lbn().at(bn_in_op);
}

void Kernel::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}
void Kernel::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

void Kernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardDataContent(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_data_id()) { ForwardDataId(ctx, BnInOp2Blob); }
}

void Kernel::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BackwardDataContent(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_data_id()) { BackwardDataId(ctx, BnInOp2Blob); }
}

void Kernel::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK_EQ(kernel_conf().input_bns().size(), 1);
  const Blob* in_blob = BnInOp2Blob(kernel_conf().input_bns(0));
  CopyDataIdToAllOb(ctx.device_ctx, BnInOp2Blob, in_blob);
}

template<DeviceType device_type>
void KernelIf<device_type>::BackwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyDataIdToAllOb(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* blob) const {
  for (const std::string& obn : kernel_conf().output_bns()) {
    Blob* output_blob = BnInOp2Blob(obn);
    output_blob->CopyDataIdFrom<device_type>(ctx, blob);
  }
}

namespace {

HashMap<int, std::function<Kernel*(const KernelConf&)>>& GetCreatorsMap() {
  static HashMap<int, std::function<Kernel*(const KernelConf&)>> obj;
  return obj;
}

}  // namespace

void AddKernelCreator(OperatorConf::OpTypeCase op_case,
                      std::function<Kernel*(const KernelConf&)> creator) {
  CHECK(GetCreatorsMap().emplace(op_case, creator).second);
}

void AddKernelCreator(OperatorConf::OpTypeCase op_case,
                      std::function<Kernel*()> creator) {
  CHECK(GetCreatorsMap()
            .emplace(op_case, [=](const KernelConf&) { return creator(); })
            .second);
}

std::unique_ptr<const Kernel> ConstructKernel(const KernelConf& conf) {
  Kernel* rptr = GetCreatorsMap().at(conf.op_conf().op_type_case())(conf);
  rptr->Init(conf);
  return std::unique_ptr<const Kernel>(rptr);
}

template class KernelIf<DeviceType::kCPU>;
template class KernelIf<DeviceType::kGPU>;

}  // namespace oneflow
