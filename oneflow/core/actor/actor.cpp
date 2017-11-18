#include "oneflow/core/actor/actor.h"

namespace oneflow {

void Actor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  actor_id_ = task_proto.task_id();
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = ConstructKernel(node.kernel_conf());
    ek.bn_in_op2regst_desc_id = PbMap2HashMap(node.bn_in_op2regst_desc_id());
    exec_kernel_vec_.push_back(std::move(ek));
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    RegstMgr::Singleton()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.emplace(pair.first, regst_desc_id).second);
  }
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.emplace(pair.first, pair.second).second);
  }
  msg_handler_ = nullptr;
  // Status of Produced Registers
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) {
      writeable_produced_regst_[regst->regst_desc_id()].push_back(regst.get());
      produced_regst2reading_cnt_[regst.get()] = 0;
    }
  }
  writeable_produced_regst_desc_num_ = writeable_produced_regst_.size();
  total_reading_cnt_ = 0;
  num_of_remaining_eord_ = -1;
  num_of_read_empty_ = -1;
  VirtualActorInit(task_proto, thread_ctx);
}

int64_t Actor::RegstDescId4Name(const std::string& name) const {
  auto find_it = name2regst_desc_id_.find(name);
  if (find_it != name2regst_desc_id_.end()) { return find_it->second; }
  return -1;
}

KernelCtx Actor::GenDefaultKernelCtx() const {
  KernelCtx ctx;
  ctx.device_ctx = device_ctx_.get();
  return ctx;
}

int Actor::HandlerZombie(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  if (total_reading_cnt_ == 0) {
    msg_handler_ = nullptr;
    return 1;
  }
  return 0;
}

void Actor::ActUntilFail() {
  while (IsReadReady() && IsWriteReady()) { Act(); }
}

bool Actor::IsWriteReady() {
  return writeable_produced_regst_desc_num_ == writeable_produced_regst_.size();
}

void Actor::ProcessOneEord() {
  num_of_remaining_eord_ -= 1;
  if (num_of_remaining_eord_ > 0) { return; }
  if (num_of_read_empty_) {
    if (!total_reading_cnt_) {
      OF_SET_MSG_HANDLER(nullptr);
    } else {
      OF_SET_MSG_HANDLER(&Actor::HandlerZombie);
    }
    AsyncSendEORDMsgForAllProducedRegstDesc();
  } else {
    OF_SET_MSG_HANDLER(&Actor::HandlerWaitUntilNoReadableRegst);
  }
}

void Actor::TrySwitchToZombie() {
  if (total_reading_cnt_ == 0) {
    OF_SET_MSG_HANDLER(nullptr);
  } else {
    OF_SET_MSG_HANDLER(&Actor::HandlerZombie);
  }
}

void Actor::AsyncLaunchKernel(
    const KernelCtx& kernel_ctx,
    std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    (ek.kernel.get()->*GetKernelWardFunc())(
        kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
          auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
          if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) {
            return nullptr;
          }
          Regst* regst = Regst4RegstDescId(regst_desc_id_it->second);
          const std::string& lbn = ek.kernel->Lbn4BnInOp(bn_in_op);
          return regst->GetBlobPtrFromLbn(lbn);
        });
  }
}

void Actor::AsyncSendRegstMsgToConsumer(
    std::function<void(Regst*)> RegstPreProcess,
    std::function<bool(int64_t)> IsAllowedActor) {
  int64_t this_actor_id = actor_id_;
  for (auto& pair : writeable_produced_regst_) {
    Regst* regst = pair.second.front();
    RegstPreProcess(regst);
    auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
    CHECK_EQ(regst_reading_cnt_it->second, 0);
    for (int64_t consumer : regst->consumers_actor_id()) {
      if (!IsAllowedActor(consumer)) { continue; }
      total_reading_cnt_ += 1;
      regst_reading_cnt_it->second += 1;
      device_ctx_->AddCallBack([consumer, regst, this_actor_id]() {
        ActorMsg msg =
            ActorMsg::BuildRegstMsgToConsumer(this_actor_id, consumer, regst);
        ActorMsgBus::Singleton()->SendMsg(std::move(msg));
      });
    }
    if (!regst->consumers_actor_id().empty()) { pair.second.pop_front(); }
    if (pair.second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
  }
}

void Actor::AsyncSendRegstMsgToConsumer(
    std::function<void(Regst*)> RegstPreProcess) {
  AsyncSendRegstMsgToConsumer(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::AsyncSendRegstMsgToConsumer(
    std::function<bool(int64_t)> IsAllowedActor) {
  AsyncSendRegstMsgToConsumer([](Regst*) {}, IsAllowedActor);
}

void Actor::AsyncSendRegstMsgToConsumer() {
  AsyncSendRegstMsgToConsumer([](Regst*) {});
}

void Actor::AsyncSendEORDMsgToConsumers(int64_t regst_desc_id) {
  const RtRegstDesc* regst_desc =
      produced_regsts_.at(regst_desc_id).front()->regst_desc();
  device_ctx_->AddCallBack([regst_desc]() {
    for (int64_t consumer : regst_desc->consumers_actor_id()) {
      ActorMsg msg = ActorMsg::BuildCommandMsg(consumer, ActorCmd::kEORD);
      ActorMsgBus::Singleton()->SendMsg(std::move(msg));
    }
  });
}

void Actor::AsyncSendEORDMsgForAllProducedRegstDesc() {
  for (const auto& pair : produced_regsts_) {
    AsyncSendEORDMsgToConsumers(pair.first);
  }
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst) {
  AsyncSendRegstMsgToProducer(regst, regst->producer_actor_id());
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst, int64_t producer) {
  ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(actor_id_, producer, regst);
  device_ctx_->AddCallBack([msg]() { ActorMsgBus::Singleton()->SendMsg(msg); });
}

void Actor::AsyncDo(std::function<void()> func) {
  device_ctx_->AddCallBack(func);
}

int Actor::TryUpdtStateAsProducedRegst(Regst* regst) {
  auto reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_regst2reading_cnt_.end()) { return -1; }
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  total_reading_cnt_ -= 1;
  if (reading_cnt_it->second != 0) { return 0; }
  auto writeable_it = writeable_produced_regst_.find(regst->regst_desc_id());
  if (writeable_it == writeable_produced_regst_.end()) { return 0; }
  if (writeable_it->second.empty()) { writeable_produced_regst_desc_num_ += 1; }
  writeable_it->second.push_back(regst);
  return 0;
}

Regst* Actor::GetCurWriteableRegst(int64_t regst_desc_id) {
  auto it = writeable_produced_regst_.find(regst_desc_id);
  if (it == writeable_produced_regst_.end()) { return nullptr; }
  return it->second.front();
}

Regst* Actor::GetCurWriteableRegst(const std::string& name) {
  return GetCurWriteableRegst(RegstDescId4Name(name));
}

Regst* Actor::GetCurSoleWriteableRegst() {
  CHECK_EQ(writeable_produced_regst_.size(), 1);
  return writeable_produced_regst_.begin()->second.front();
}

}  // namespace oneflow
