namespace RS
open Hopac
open System
open System.Collections.Concurrent
open System.Collections.Generic
open MathNet.Numerics.Statistics
open System.Collections.Generic
open MathNet.Numerics.LinearAlgebra

type FPLV_type=
    /// FP GV all possible neighbors
    |FPLV_All
    /// FP GV top k neariest neighbors
    |FPLV_KNN
    /// FP GV all possible neighbors with similarity selection
    |FPLV_All_SS 
    // =FP_GV_All + community infomation with similarity selection
    |FPLV_C // the best one.
    

module FPLV= 
    let long_name = "2023 Plos One"    
    let short_name (gv_type:FPLV_type)=
        match gv_type with 
        | FPLV_All -> "FPLV-ALL"
        | FPLV_All_SS -> "FPLV-SS"
        | FPLV_KNN -> "FPLV-KNN"
        | FPLV_C -> "FPLV-C"        
    module Types =
        type User_Rating_Min_Max=
            {
                //UserId:int<UserId>
                MaxRating:float32
                MinRating:float32
                Average:float32
            }
        type FuzzyPreference=
            {    
                Like :float32
                Dislike :float32  
            }
    module Base = 
        open Types
        let inline preference (min_: float32) (max_: float32) (avg_: float32) (v: float32) =
            if v >= avg_ then
                let like = 
                    if max_ = avg_ then 1.0f
                    else
                        (v - avg_) / (max_ - avg_)
                {
                    Like = like
                    Dislike = 0.0f
                }
            else
                let dislike =
                    if min_ = avg_ then 0.0f
                    else
                        (avg_ - v) / (avg_ - min_)
                {
                    Like = 0.0f
                    Dislike = dislike
                }
        ///1.获取每个用户的FP特征，实际上相当于把评分转换了。
        let inline get_FP_features 
            (user_min_max_rating:IDictionary<int<UserId>,User_Rating_Min_Max>)
            (ratings:Rating seq)
            :IDictionary<int<UserId>*int<ItemId>,FuzzyPreference>
            =
            ratings
            |>Seq.toArray
            |>Array.Parallel.map (fun r ->
                let profile=user_min_max_rating[r.UserId]
                let fp=preference profile.MinRating profile.MaxRating profile.Average r.Rating
                (r.UserId,r.ItemId),fp
                )
            |>dict
        ///2.已知两个用户FP特征，计算两个用户的相似性。
        let inline get_local_fuzzy_similarity (similarity_function) (user_fp:seq<FuzzyPreference*FuzzyPreference>)  
            =
            if (Seq.length user_fp) =0
            then None
            else
                let get_likes= fun fp -> fp.Like
                let get_dislikes= fun fp -> fp.Dislike
                let user_1_fp,user_2_fp=user_fp|>Seq.toArray|>Array.unzip
                let likes_by_u1 = user_1_fp|>Seq.map get_likes
                let likes_by_u2 = user_2_fp|>Seq.map get_likes
                let dislikes_by_u1 = user_1_fp|>Seq.map get_dislikes
                let dislikes_by_u2 = user_2_fp|>Seq.map get_dislikes
                let s_like = similarity_function likes_by_u1 likes_by_u2
                let s_dis = similarity_function dislikes_by_u1 dislikes_by_u2        
                match s_like,s_dis with
                | Some l,Some d ->
                    (l + d) / 2.0
                    |>Some
                | _-> None
        // 已知相似性向量，求相似性标量，公式9
        let vec_to_similarity v =
            v
            |>Seq.map (fun dev ->
                dev*dev
            )    
            |>Seq.sum|>sqrt
        let get_average_similarity_via_gv (possible_max:float) (shared_ratings:(float*float*Vector<float>) seq)=
            if Seq.length(shared_ratings)=0
            then None
            else
                let s2=
                    shared_ratings 
                    |>Seq.map (fun (r1,r2,v) ->
                        let s=possible_max - abs(r1 - r2)
                        v*s
                    )   
                    |>Seq.fold (fun s t  -> s + t) (GenreVector.emptyVector())
                    |>Seq.toArray
                let s3=
                    shared_ratings 
                    |>Seq.map (fun (r1,r2,v) ->
                        v 
                    )   
                    |>Seq.fold (fun s t  -> s + t) (GenreVector.emptyVector())
                    |>Seq.toArray
                let r=
                    s3
                    |>Array.mapi (fun index el ->
                        if el = 0.0
                        then 0.0 
                        else 
                            s2[index] / el
                            //|>Some                
                        )
                Some r
        //---
        
        
        let inline private user_ave (trained:ITrained) u=
            if trained.DataCut.User_Average_Rating.ContainsKey u 
            then trained.DataCut.User_Average_Rating[u] 
            else trained.DataCut.Average_rating_in_train 
        let inline private item_ave (trained:ITrained) i=
            if trained.DataCut.Item_Average_Rating.ContainsKey i 
            then trained.DataCut.Item_Average_Rating[i] 
            else trained.DataCut.Average_rating_in_train 
        /// predict r by 'target_user_average_rating', neighbors 'rating','average rating' and 'similarity'
        let ``predict r by absolute similarity and average rating``
            (target_user_average_rating:float32)
            (neighbors:seq<CFPredictionDataWithAve>)
            =    
                if Seq.length(neighbors)=0
                then target_user_average_rating 
                else
                    let numerator =
                        neighbors                
                        |> Seq.sumBy (fun n->  (n.NeighborRating - n.NeighborAverageRating)*n.NeighborSimilarity )                
                    let denominator = neighbors |> Seq.sumBy (fun (n)-> n.NeighborSimilarity |> abs) // here abs(similarity) is used 
                    if  (Float.equal32 1.0e-10f (float32  denominator) 0.0f) //float32(denominator)=0.0f
                    then target_user_average_rating
                    else
                        Float.tryNan32(numerator/denominator)
                        |>Option.map (fun r -> target_user_average_rating + r )
                        |>Option.defaultValue target_user_average_rating
        /// predict r by 'target_user_average_rating', neighbors 'rating','average rating' and 'similarity'
        let ``absolute similarity with average rating and node community``
            (node_community:IDictionary<int,int>)
            (weight_for_nodes_in_same_community:float32)
            (weight_for_nodes_in_different_community:float32)
            (target_id:int)
            (target_user_average_rating:float32)
            (neighbors:seq<CFPredictionDataWithAve>)
            =    
                let checkFunction target_id neighbor_id =
                    if node_community[target_id]=node_community[neighbor_id]
                    then weight_for_nodes_in_same_community
                    else weight_for_nodes_in_different_community
                if Seq.length(neighbors)=0
                then target_user_average_rating 
                else
                    let numerator =
                        neighbors                
                        |> Seq.sumBy (fun (n)->
                            let w=checkFunction target_id n.Neighbor_Id
                            (n.NeighborRating - n.NeighborAverageRating)*n.NeighborSimilarity * w)                
                    let denominator = neighbors |> Seq.sumBy (fun (n)-> n.NeighborSimilarity |> abs) // here abs(similarity) is used 
                    if  (Float.equal32 1.0e-10f (float32 denominator) 0.0f) //float32(denominator)=0.0f
                    then target_user_average_rating
                    else
                        Float.tryNan32(numerator/denominator)
                        |>Option.map (fun r -> target_user_average_rating + r )
                        |>Option.defaultValue target_user_average_rating
        let inline try_cache (cache : Dictionary<int<Id>,Dictionary<int<Id>,float32>>) (targets:int*int)= 
            let a,b=targets
            let a= a*1<Id>
            let b= b*1<Id>
            if cache.ContainsKey(a) && cache[a].ContainsKey(b)
            then 
                Some cache.[a].[b]
            else None
        let inline try_in_or_default (user:int<Id>) (cache : IDictionary<int<UserId>,float32>) (default_val:float32) =
            let u = user|>int |>fun i -> i*1<UserId>
            match cache.ContainsKey(u) with
            |true -> cache[u]
            |false -> default_val
    module Components =        
        open Types
        let getUser_Rating_Min_Max 
            (par:Map<Parameters,obj>) 
            (data_cut:IDataCut)
            :Job<IDictionary<int<UserId>,User_Rating_Min_Max>> 
            =
            job{
                let r=
                    data_cut.Users
                    |>Seq.toArray
                    |>Array.Parallel.map (fun user ->
                        let ratings = 
                            if data_cut.RawData.UserDict.ContainsKey user
                            then
                                data_cut.RawData.UserDict[user]
                                |>Seq.choose data_cut.rating_in_training_set_by_index
                            else Seq.empty
                        if Seq.length(ratings)=0
                        then 
                            let r=data_cut.Average_rating_in_train
                            user,{MaxRating=5.0f;MinRating=0.5f;Average=r}
                        else
                            let max_=ratings|>Seq.maxBy(fun r -> r.Rating)
                            let min_=ratings|>Seq.minBy(fun r -> r.Rating)
                            let ave=data_cut.User_Average_Rating[user]
                            user,{MaxRating=max_.Rating;MinRating=min_.Rating;Average=ave}    
                        )
                    |>dict        
                return r
            }
        
        let getFuzzyPreference 
            user_max_min_rating
            (par:Map<Parameters,obj>) 
            (data_cut:IDataCut)
            :Job<IDictionary<int<UserId>*int<ItemId>,FuzzyPreference>> 
            =
            job{
                let all_rating = 
                    data_cut.all_rating_in_training_set()
                let r= Base.get_FP_features user_max_min_rating all_rating   
                return r
            }
        
        let getPearsonSimilarity 
            (par:Map<Parameters,obj>) 
            (data_cut:IDataCut)
            (threshold:float32)        
            =        
            job{
                let inline pearson_local u v= SimilarityFunctions.pearson2 u v                 
                let cf_type:CFType=
                    par
                    |>Map.get_or_default<_,CFType> ``Collabrative filtering type: UserBased, ItemBased`` (CFType.UserBased)                           
                let! result=Pearson.getSimilarity par data_cut PearsonParameters.``Take positive PCC only`` (SimilarityFunctions.pearson)                    
                return result
            }
        let get_both_vectorSimilarity_and_scalarSimilarity     
            (fuzzy_preference:IDictionary<int<UserId>*int<ItemId>,FuzzyPreference>)
            (item_gv:IDictionary<int<ItemId>,VectorContainer<Genre>>)
            (gv_type:FPLV_type)            
            (par:Map<Parameters,obj>) 
            (data_cut:IDataCut)
            =
            job {        
                let job_name= $"Feature:like dislike vector Pearson-%A{data_cut.Id}"               
                logger.Information $"%s{job_name} start"
                let cf_type:CFType=
                    par
                    |>Map.get_or_default<_,CFType> ``Collabrative filtering type: UserBased, ItemBased`` (CFType.UserBased)
                                
                let! pearson_larger_than_threhold =
                    match gv_type with
                    | FPLV_All_SS |FPLV_C->
                        Pearson.getSimilarity par data_cut PearsonParameters.``Take positive PCC only`` (SimilarityFunctions.pearson) 
                    |_->
                        let cache_type:CacheType=
                            par
                            |>Map.get_or_default<_,CacheType> ``Cache type: CacheType`` (CacheType.CacheInFile)
                        let result_cache = 
                            match cache_type with
                            | CacheType.CacheInFile ->
                                Cache.createInFile<IDictionary<int<Id>,float32>> $"{short_name}_{data_cut.Id}_similarity"  
                            | CacheType.CacheInMemory ->
                                Cache.createInMemory<IDictionary<int<Id>,float32>>()
                        result_cache|>Job.result
                let cache_type:CacheType=
                    par
                    |>Map.get_or_default<_,CacheType> ``Cache type: CacheType`` (CacheType.CacheInFile)
                let vs_cache = 
                    match cache_type with
                    | CacheType.CacheInFile ->
                        Cache.createInFile<IDictionary<int<Id>,UserGenreVectorSimilarityPearson>> $"{short_name}_{data_cut.Id}_similarity"  
                    | CacheType.CacheInMemory ->
                        Cache.createInMemory<IDictionary<int<Id>,UserGenreVectorSimilarityPearson>>()
                let scalar_cache = 
                    match cache_type with
                    | CacheType.CacheInFile ->
                        Cache.createInFile<IDictionary<int<Id>,float32>> $"{short_name}_{data_cut.Id}_similarity"  
                    | CacheType.CacheInMemory ->
                        Cache.createInMemory<IDictionary<int<Id>,float32>>()
                let vs_cache_local : Dictionary<int<Id>,Dictionary<int<Id>,UserGenreVectorSimilarityPearson>> = Dictionary()
                let scalar_cache_local : Dictionary<int<Id>,Dictionary<int<Id>,float32>> = Dictionary()
                let all_pairs = 
                    match cf_type with 
                    |UserBased ->data_cut.Users|>Seq.map (int)
                    |ItemBased ->data_cut.Items.Keys|>Seq.map (int) 
                    |>Seq.map (fun i ->
                        vs_cache_local.Add(i*1<Id>,Dictionary())
                        scalar_cache_local.Add(i*1<Id>,Dictionary())
                        i
                        )
                    |>Pair.undirected2
                for (target1,neighbors) in all_pairs do
                    let filtered_pairs=
                        neighbors              
                        |>Array.choose (fun (target2) ->  
                            match Base.try_cache scalar_cache_local (target1,target2) with
                            |None ->
                                Some(target1*1<Id>,target2*1<Id>)                                     
                            |Some x -> 
                                None
                            )            
                    let vector_sim_result =
                        filtered_pairs
                        |>Array.Parallel.choose (fun (target1,target2)->
                            let user1 = target1|>int|>fun i->i*1<UserId>
                            let user2 = target2|>int|>fun i->i*1<UserId>
                            let shared_ratings=
                                data_cut.shareTargets cf_type (target1,target2 )
                                |>Seq.map (fun r ->                             
                                    r.Rating1,r.Rating2,r.SharedTarget|>int|>fun i-> i*1<ItemId>
                                    )  
                            let gv_data_like,gv_data_dislike=
                                shared_ratings
                                |>Seq.choose (fun s->
                                    let (rui,rvi,item) = s
                                    let gv = item_gv[item]
                                    let gv2= GenreVector.cloneGenreVector gv
                                    let u1i=fuzzy_preference.[user1,item]
                                    let u2i=fuzzy_preference.[user2,item]
                                    let likes=float(u1i.Like),float(u2i.Like),gv.Vec
                                    let dislikes=float(u1i.Dislike),float(u2i.Dislike),gv2.Vec
                                    Some(likes,dislikes)                    
                                    )
                                |>Seq.toArray
                                |>Array.unzip
                            let sim_like,sim_dislike=
                                Base.get_average_similarity_via_gv 1.0 gv_data_like,Base.get_average_similarity_via_gv 1.0 gv_data_dislike                       
                            let r=
                                match sim_like,sim_dislike with 
                                | Some s_like,Some s_dislike ->
                                    {
                                        User1=user1
                                        User2=user2
                                        s_ij_like=s_like
                                        s_ij_dislike=s_dislike
                                    }|>Some 
                                | _ -> None
                            match gv_type with
                            |FPLV_All | FPLV_KNN -> r
                            |FPLV_All_SS | FPLV_C ->
                                match (pearson_larger_than_threhold.TryRetrieve(string(user1))) with
                                |Some d -> 
                                    let u2 = user2|>int|>fun i ->i*1<Id>
                                    if d.ContainsKey(u2)
                                    then r else None
                                |None ->None
                        )
                    let scalar_sim_result =
                        vector_sim_result
                        |>Array.Parallel.map (fun (result)->                    
                            let s_like = result.s_ij_like |>Base.vec_to_similarity
                            let s_dislike = result.s_ij_dislike |>Base.vec_to_similarity
                            let sim = (s_like + s_dislike)/2.0 |>float32
                            let u1=int (result.User1)*1<Id>
                            let u2=int (result.User2)*1<Id>                    
                            u1,u2,sim
                            )
                    ()
                    // cache the results.
                    vector_sim_result
                    |>Array.iter (fun (vs)->
                        let u = vs.User1|>int|>fun i ->i*1<Id>
                        let v = vs.User2|>int|>fun i ->i*1<Id>
                        vs_cache_local[u].TryAdd(v,vs)|>ignore
                        vs_cache_local[v].TryAdd(u,vs)|>ignore
                        )                 
                    scalar_sim_result
                    |>Array.iter (fun (u,v,s)->                        
                        scalar_cache_local[u].TryAdd(v,s)|>ignore
                        scalar_cache_local[v].TryAdd(u,s)|>ignore
                        ) 
                    
                    scalar_cache.Add (string(target1),scalar_cache_local[target1*1<Id>])
                    vs_cache.Add (string(target1),vs_cache_local[target1*1<Id>])
                let scalar_cache_local2=
                    scalar_cache_local
                    |>Seq.map (fun kv ->
                        kv.Value
                        |>Seq.map (fun kv2 -> kv2.Key,kv2.Value)
                        |>dict
                        |>fun d -> kv.Key,d
                        )
                    |>dict
                let result=
                    {|
                        Vector_similarity=vs_cache
                        Scalar_similarity=scalar_cache
                        Scalar_similarity_local=scalar_cache_local2
                    |}
                vs_cache_local.Clear()
                scalar_cache_local.Clear()
                return result
            }
        let make_similarity_network 
            (similarity:IDictionary<int<Id>,IDictionary<int<Id>,float32>>) 
            (par:Map<Parameters,obj>)
            (data_cut:IDataCut) 
            = 
            logger.Information $"{data_cut.Id}:similarity network modeling starts..."
            let cf_type:CFType=
                par
                |>Map.get_or_default<_,CFType> ``Collabrative filtering type: UserBased, ItemBased`` (CFType.UserBased)
            let node_label =
                match cf_type with
                |UserBased -> data_cut.Users |>Seq.map (fun u -> int u,string u)|>dict
                |ItemBased->data_cut.Items|>Seq.map (fun u -> int u.Key, u.Value.Name)|>dict
            let MinimumConnection=
                par
                |>Map.get_or_default<_,int> Parameters.``Minimum connection of similarity network: int`` 1 
                 
            let average_degree=
                par
                |>Map.get_or_default<_,float> Parameters.``Average degree of similarity network: float`` 1.5 
                            
            let IsDirected=
                par
                |>Map.get_or_default<_,bool> Parameters.``Is similarity network directed: bool`` false
                 
            logger.Information $"{data_cut.Id}:similarity network is using MinimumConnection={MinimumConnection},AverageDegree={average_degree},IsDirected={IsDirected}"        
            let net= Network.makeSimilarityNetwork2 similarity (Some node_label) MinimumConnection average_degree IsDirected            
            net
        let get_neighbor_by_user_item 
            (trained:ITrained)            
            (user:int<UserId>,item:int<ItemId>)
            =
            let par = trained.Parameters
            let data_cut =trained.DataCut
            let cf_type=            
                Map.get_or_default<_,CFType> ``Collabrative filtering type: UserBased, ItemBased`` CFType.UserBased par            
            let sim=
                trained.Results
                |>ConcurrentDictionary.getAsOrFail<IDictionary<int<Id>,IDictionary<int<Id>,UserGenreVectorSimilarityPearson>>> "similarity"
            let user2=int(user)*1<Id>       
            match sim.ContainsKey(user2) with
            | false ->                 
                logger.Debug $"{user}->{item}, doesn't have similarity."
                (user,item),Array.empty               
            | true ->
                sim[user2]
                |> Seq.toArray
                |> Array.choose (fun kv ->                    
                    let cell= kv.Value
                    let a,b=
                        match cf_type with
                        | ItemBased -> user,kv.Key|>fun i -> int(i)*1<ItemId>
                        | UserBased -> kv.Key|>fun j -> int(j)*1<UserId>,item
                    match data_cut.rating_in_training_set (a,b) with
                    | Some rating ->                         
                        (a,b,cell)|>Some
                    |None ->None 
                    )            
                |> Array.Parallel.map (fun (a,b,r) ->
                    match cf_type with
                        | ItemBased -> int b|>fun i -> i*1<Id>,r
                        | UserBased -> int a|>fun i -> i*1<Id>,r
                    )   
                |>fun n -> (user,item),n
     
    
    let train
        (p:FPLV_type)
        (par:Map<Parameters,obj>) 
        (data_cut:IDataCut) :Job<ITrained> 
        = 
        job{
            let s_name =short_name p
            let exp_id=data_cut.Id            
            let ts = DateTime.Now
            let trained_results = ConcurrentDictionary<string,obj>()   
            //-----
            logger.Information $"{s_name} ex_{exp_id}:training starts..." 
            logger.Information $"{s_name} ex_{exp_id}:Entropy ended..." 
            logger.Information $"{s_name} ex_{exp_id}:Item GenreVector starts..."
            let r_GenreVector = VectorSimilarity.Components.get_GenreVector par data_cut
            trained_results.TryAdd("Item GenreVector",r_GenreVector)|>ignore 
            logger.Information $"{s_name} ex_{exp_id}:Item GenreVector ended..." 
            logger.Information $"{s_name}-{exp_id}:user max and min rating starts..." 
            let! user_max_and_min = Components.getUser_Rating_Min_Max par data_cut                        
            trained_results.TryAdd("user max and min rating",user_max_and_min)|>ignore
            logger.Information $"{s_name}-{exp_id}:user max and min rating done..." 
            logger.Information $"{s_name}-{exp_id}:fuzzy preference start..." 
            let! fp=Components.getFuzzyPreference user_max_and_min par data_cut
            trained_results.TryAdd("fuzzy preference",fp)|>ignore
            logger.Information $"{s_name} ex_{exp_id}:Similarity starts..." 
            let! sim=Components.get_both_vectorSimilarity_and_scalarSimilarity fp r_GenreVector p par data_cut                 
            trained_results.TryAdd("vector similarity",sim.Vector_similarity)|>ignore
            trained_results.TryAdd("scalar similarity",sim.Scalar_similarity)|>ignore
            logger.Information $"{s_name} ex_{exp_id}:similarity saved... "
            match p with            
            |FPLV_C->
                logger.Information $"{s_name}-{exp_id}:SimilarityNetwork calculating..."
                let net = Components.make_similarity_network sim.Scalar_similarity_local par data_cut                  
                let community,k_core=KCore.get_KCore_community net
                let community=community|>Seq.map (fun kv-> int(kv.Key) ,int(kv.Value))|>dict
                trained_results.TryAdd("community result",community)|>ignore
                logger.Information $"{s_name}-{exp_id}:Community info added..."
            | _ -> ()   
            Validate.record_time trained_results { Experment_Id=exp_id;Algorithm=s_name;Event=TimeEvent.Training;TimeSpan= (DateTime.Now - ts).TotalMinutes}
            logger.Information $"{s_name} ex_{exp_id}:training done..." 
            let trained_model =
                {
                    new ITrained with
                        member t.DataCut=data_cut
                        member t.Parameters=par
                        member t.Results=trained_results
                }
            return trained_model
        }
          
        
    
    
    let test (p:FPLV_type) (trained:ITrained) :Job<Observation_of_Prediction []>= 
        job{
            let short_name =short_name p
            let exp_id=trained.DataCut.Id
            let ts = DateTime.Now
            logger.Information $"{short_name} ex_{exp_id}:testing starts..."
            let node_community = 
                trained.Results|>ConcurrentDictionary.tryGet<IDictionary<int,int>> "community result"
            
            let get_neighbors =Components.get_neighbor_by_user_item trained
            let ob = 
                let item_gv,vs,similarity =
                    Picker.pick3_or_fail
                        (fun _ -> 
                            trained.Results
                            |>ConcurrentDictionary.tryGet<IDictionary<int<ItemId>,VectorContainer<Genre>>> "Item GenreVector"
                            |>fun r ->
                                match r with
                                |Some r -> Ok r 
                                |None -> Error "failed to read 'Item GenreVector'"
                            )
                        (fun _ -> 
                            trained.Results
                            |>ConcurrentDictionary.tryGet<ICache<string,IDictionary<int<Id>,UserGenreVectorSimilarityPearson>>> "vector similarity"
                            |>fun r ->
                                match r with
                                |Some r -> Ok r 
                                |None -> Error "failed to read 'vector similarity'"
                            )
                        (fun _ -> 
                            trained.Results
                            |>ConcurrentDictionary.tryGet<ICache<string,IDictionary<int<Id>,float32>>> "scalar similarity"
                            |>fun r ->
                                match r with
                                |Some r -> Ok r 
                                |None -> Error "failed to read 'scalar similarity'"
                            )
                     
                let cf_type:CFType=
                    trained.Parameters                
                    |>Map.tryGet ``Collabrative filtering type: UserBased, ItemBased`` //(CFType.UserBased)
                    |>fun r -> defaultArg r CFType.UserBased
                //[1..19]@[20..2..59]@[60..5..200]
                let neighbor_list=
                    Map.get_or_default<_,int list> ``Required number of neighbors: int list`` ([1..19]@[20..2..59]@[60..5..200]) trained.Parameters                
                    |>Array.ofList
                logger.Information $"{short_name} ex_{exp_id}:get the testing set..."
                let testing =trained.DataCut.Test_set
                testing
                |>Seq.map (fun r ->
                    let r1=
                        match trained.DataCut.RawData.get_rating r with
                        |Some rating -> rating 
                        |_-> 
                            logger.Fatal "A rating in testing is not contained in the data set. this should not happen."
                            failwith "A rating in testing is not contained in the data set. this should not happen."
                    r1
                )
                |>Seq.groupBy (fun r -> r.UserId) //按用户分组
                |>Seq.toArray
                |>Array.Parallel.collect (fun (u,data)->
                    let file_key = u |>int |>string
                    let neigh =                                 
                        similarity.TryRetrieve file_key
                        |>fun d -> Option.defaultValue ([]|>dict) d
                    data
                    |>Array.ofSeq
                    |>Array.map (fun (target_rating:Rating) ->                            
                        let u_i = target_rating.UserId,target_rating.ItemId
                        let tar:int<Id>= 
                            if cf_type=UserBased 
                            then fst u_i |>int
                            else snd u_i |>int
                            |>fun i -> i*1<Id>
                        let tar_ave =DataCut.target_ave_rating trained.DataCut cf_type tar
                        let all_neighbor_qualified = 
                            neigh
                            |>Seq.toArray
                            |>Array.choose (fun kv ->
                                let (id_,similarity) = kv.Key,kv.Value
                                let r =DataCut.has_rated_target trained.DataCut cf_type u_i id_
                                r
                                |>Option.map (fun rating ->
                                    {
                                        Neighbor_Id=id_|>int
                                        NeighborRating=rating.Rating
                                        NeighborAverageRating=DataCut.target_ave_rating trained.DataCut cf_type id_
                                        NeighborSimilarity=similarity
                                    }
                                    )
                            )
                            |>Array.sortByDescending (fun data -> data.NeighborSimilarity)
                        let results=
                            neighbor_list
                            |>Array.Parallel.map (fun required_neighbor ->
                                let neighbors=
                                    match p with 
                                    | FPLV_KNN ->
                                        match all_neighbor_qualified.Length>=required_neighbor with 
                                        |true -> Seq.take required_neighbor all_neighbor_qualified
                                        |_->all_neighbor_qualified
                                    |_->all_neighbor_qualified
                                let used_neighbors = Seq.length neighbors
                                //used_neighbors,Predict.``absolute similarity with average rating`` ave neighbors_info 
                                let predicted= 
                                    match node_community with
                                    |Some community_info ->                                
                                        //logger.Information $"{s_name}-{exp_id}:predict with community info..."
                                        Base.``absolute similarity with average rating and node community``                                         
                                            community_info
                                            1.0f
                                            0.7f
                                            (int tar)
                                            tar_ave
                                            neighbors
                                    |None ->                                
                                        Base.``predict r by absolute similarity and average rating`` tar_ave neighbors 
                                let ob=
                                    {
                                        Algorithm=short_name
                                        User=target_rating.UserId
                                        Item=target_rating.ItemId
                                        Required_neighbor = required_neighbor
                                        Used_neighbor =used_neighbors
                                        Predicted=predicted
                                        Actual=target_rating.Rating                                            
                                    }
                                ob
                            )
                        results
                    )
                )
                |>Array.concat
                
            logger.Information $"{short_name} ex_{exp_id}:get the prediction observation..."
            Validate.record_time trained.Results { Experment_Id=exp_id;Algorithm=short_name;Event=TimeEvent.Test;TimeSpan= (DateTime.Now - ts).TotalMinutes}
            logger.Information $"{short_name} ex_{exp_id}:testing end..."
            return ob
        }            
    let predict (p:FPLV_type) (trained:ITrained) (targets:seq< int<UserId>*int<ItemId> >) :Job<Prediction seq * RecommendationList seq> = 
        job{
            let s_name =short_name p
            let exp_id=trained.DataCut.Id
            let ts = DateTime.Now
            logger.Information $"{s_name} ex_{exp_id}:prediction starts..."
            let node_community = 
                trained.Results|>ConcurrentDictionary.tryGet<IDictionary<int,int>> "community result"
            let predictions = 
                let item_gv,vs,similarity =
                    Picker.pick3_or_fail
                        (fun _ -> 
                            trained.Results
                            |>ConcurrentDictionary.tryGet<IDictionary<int<ItemId>,VectorContainer<Genre>>> "Item GenreVector"
                            |>fun r ->
                                match r with
                                |Some r -> Ok r 
                                |None -> Error "failed to read 'Item GenreVector'"
                            )
                        (fun _ -> 
                            trained.Results
                            |>ConcurrentDictionary.tryGet<ICache<string,IDictionary<int<Id>,UserGenreVectorSimilarityPearson>>> "vector similarity"
                            |>fun r ->
                                match r with
                                |Some r -> Ok r 
                                |None -> Error "failed to read 'vector similarity'"
                            )
                        (fun _ -> 
                            trained.Results
                            |>ConcurrentDictionary.tryGet<ICache<string,IDictionary<int<Id>,float32>>> "scalar similarity"
                            |>fun r ->
                                match r with
                                |Some r -> Ok r 
                                |None -> Error "failed to read 'scalar similarity'"
                            )
                 
                let cf_type:CFType=
                    trained.Parameters                
                    |>Map.tryGet ``Collabrative filtering type: UserBased, ItemBased`` //(CFType.UserBased)
                    |>fun r -> defaultArg r CFType.UserBased
                //[1..19]@[20..2..59]@[60..5..200]
                let neighbor_parameter=
                    Map.get_or_default<_,int> ``Optimal number of neighbors: int`` 40 trained.Parameters 
                logger.Information $"{s_name} ex_{exp_id}:using %A{neighbor_parameter} neighbors in prediction..."
                logger.Information $"{s_name} ex_{exp_id}:get the testing set..."                        
                targets                        
                |>Seq.groupBy (fun (u,i) -> u) //按用户分组
                |>Seq.toArray
                |>Array.Parallel.collect (fun (u,target_user_item)->
                    let file_key = u |>int |>string
                    let neigh =                                 
                        similarity.TryRetrieve file_key
                        |>fun d -> Option.defaultValue ([]|>dict) d
                    target_user_item
                    |>Array.ofSeq
                    |>Array.map (fun (u,i)->                            
                        let u_i = (u,i)
                        let tar:int<Id>= 
                            if cf_type=UserBased 
                            then fst u_i |>int
                            else snd u_i |>int
                            |>fun i -> i*1<Id>
                        let tar_ave =DataCut.target_ave_rating trained.DataCut cf_type tar
                        let all_neighbor_qualified = 
                            neigh
                            |>Seq.toArray
                            |>Array.choose (fun kv ->
                                let (id_,similarity) = kv.Key,kv.Value
                                let r =DataCut.has_rated_target trained.DataCut cf_type u_i id_
                                r
                                |>Option.map (fun rating ->
                                    {
                                        Neighbor_Id=id_|>int
                                        NeighborRating=rating.Rating
                                        NeighborAverageRating=DataCut.target_ave_rating trained.DataCut cf_type id_
                                        NeighborSimilarity=similarity
                                    }
                                    )
                            )
                            |>Array.sortByDescending (fun data -> data.NeighborSimilarity)
                        let results=                             
                            let neighbors=
                                match p with 
                                | FPLV_KNN ->
                                    match all_neighbor_qualified.Length>=neighbor_parameter with 
                                    |true -> Seq.take neighbor_parameter all_neighbor_qualified
                                    |_->all_neighbor_qualified
                                |_->all_neighbor_qualified
                            let used_neighbors = Seq.length neighbors
                            //used_neighbors,Predict.``absolute similarity with average rating`` ave neighbors_info 
                            let predicted= 
                                match node_community with
                                |Some community_info ->                                
                                    //logger.Information $"{s_name}-{exp_id}:predict with community info..."
                                    Base.``absolute similarity with average rating and node community``                                         
                                        community_info
                                        1.0f
                                        0.7f
                                        (int tar)
                                        tar_ave
                                        neighbors
                                |None ->                                
                                    Base.``predict r by absolute similarity and average rating`` tar_ave neighbors 
                            let ob=
                                {
                                    UserId=u
                                    ItemId=i
                                    ActualRating=None  //测试集有此数据，实际应用无
                                    PredictedRating=predicted
                                }
                            ob                             
                        results                        
                    )
                )
                        //|>fun obs ->
                        //    Frame.ofRecords obs
            logger.Information $"{s_name} ex_{exp_id}:making Recommendation..."
            let recommendation_list=Recommendation.make trained.DataCut.Average_rating_in_train predictions
            logger.Information $"{s_name} ex_{exp_id}:Recommendation done"
            logger.Information $"{s_name} ex_{exp_id}:get the prediction observation..."
            Validate.record_time trained.Results { Experment_Id=exp_id;Algorithm=s_name;Event=TimeEvent.Prediction;TimeSpan= (DateTime.Now - ts).TotalMinutes}
            logger.Information $"{s_name} ex_{exp_id}:prediction end..."
            return (predictions,recommendation_list)
        }
    let make (p:FPLV_type) =
        {
            new IRecAlgo with
                member t.FullName = short_name(p) 
                member t.ShortName  = long_name
                member t.train  par_all data_cut  = train p par_all data_cut
                member t.test trained = test p trained
                member t.predict trained targets= predict p trained targets
        }


